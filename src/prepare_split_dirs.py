"""Hardlink images+masks into per-split folders (train/val/test) as expected
by LightlyTrain's train_semantic_segmentation data format. Uses hardlinks
(same NTFS volume) instead of copies - zero extra disk usage, near-instant.
"""
import os
import pandas as pd

RAW_IMAGES_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection\train_images"
MASKS_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\masks"
SPLIT_IDS_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"

SPLITS = ["train", "val", "test"]


def hardlink_or_skip(src, dst):
    if os.path.exists(dst):
        return
    os.link(src, dst)


def main():
    for split in SPLITS:
        ids_path = f"{SPLIT_IDS_DIR}/{split}_ids.csv"
        ids = pd.read_csv(ids_path)["ImageId"].tolist()

        img_out_dir = f"{OUT_DIR}/images/{split}"
        mask_out_dir = f"{OUT_DIR}/masks_split/{split}"
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)

        for image_id in ids:
            stem = os.path.splitext(image_id)[0]
            hardlink_or_skip(f"{RAW_IMAGES_DIR}/{image_id}", f"{img_out_dir}/{image_id}")
            hardlink_or_skip(f"{MASKS_DIR}/{stem}.png", f"{mask_out_dir}/{stem}.png")

        print(f"{split}: {len(ids)} images/masks linked -> {img_out_dir}, {mask_out_dir}")


if __name__ == "__main__":
    main()
