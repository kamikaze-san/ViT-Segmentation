"""Oversample rare-class training images via extra hardlinks (filename-based,
not index-based - avoids any risk of misaligning weights to the wrong image
that a custom sampler would risk given LightlyTrain's unsorted os.walk +
mmap-cached internal file listing).

Only adds copies of images whose rarest present class is 1, 2, or 4 - leaves
background-only and class-3-only images untouched (single copy, same as
before). Val/test are never touched.
"""
import os
import pandas as pd

DATA_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
BASE_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"

# Duplication factor per rarest-class-in-image (1 = no extra copies, i.e. keep as-is)
DUP_FACTOR = {1: 3, 2: 10, 3: 1, 4: 3, "bg": 1}


def hardlink_or_skip(src, dst):
    if os.path.exists(dst):
        return
    os.link(src, dst)


def main():
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    train_ids = pd.read_csv(f"{BASE_DIR}/train_ids.csv")["ImageId"].tolist()
    classes_per_image = df[df.ImageId.isin(train_ids)].groupby("ImageId")["ClassId"].apply(set).to_dict()

    img_src_dir = f"{BASE_DIR}/images/train"
    mask_src_dir = f"{BASE_DIR}/masks_split/train"
    img_out_dir = f"{BASE_DIR}/images/train_oversampled"
    mask_out_dir = f"{BASE_DIR}/masks_split/train_oversampled"
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    dup_counts = {1: 0, 2: 0, 3: 0, 4: 0, "bg": 0}
    total_written = 0

    for image_id in train_ids:
        stem = os.path.splitext(image_id)[0]
        classes = classes_per_image.get(image_id)
        rarest = max(classes, key=lambda c: DUP_FACTOR[c]) if classes else "bg"
        factor = DUP_FACTOR[rarest]
        dup_counts[rarest] += 1

        src_img = f"{img_src_dir}/{image_id}"
        src_mask = f"{mask_src_dir}/{stem}.png"

        for copy_idx in range(factor):
            suffix = "" if copy_idx == 0 else f"__dup{copy_idx}"
            hardlink_or_skip(src_img, f"{img_out_dir}/{stem}{suffix}.jpg")
            hardlink_or_skip(src_mask, f"{mask_out_dir}/{stem}{suffix}.png")
            total_written += 1

    print("Source images by rarest present class:")
    for k, v in dup_counts.items():
        print(f"  {k}: {v} images -> {v * DUP_FACTOR[k]} copies (factor {DUP_FACTOR[k]}x)")
    print(f"\nTotal images in oversampled train set: {total_written} (was {len(train_ids)})")


if __name__ == "__main__":
    main()
