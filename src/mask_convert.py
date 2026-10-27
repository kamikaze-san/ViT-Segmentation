"""RLE -> single-channel class-index PNG masks for every image in
train_images (values 0=background, 1-4=defect class). Confirmed earlier
that defect classes never overlap at the pixel level, so a single channel
is sufficient - no need for multi-channel masks.
"""
import os
import cv2
import numpy as np
import pandas as pd

from rle_utils import rle_decode

DATA_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\masks"
IMG_SHAPE = (256, 1600)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    grouped = df.groupby("ImageId")

    all_image_ids = sorted(os.listdir(f"{DATA_DIR}/train_images"))
    print(f"Building masks for {len(all_image_ids)} images ...")

    for i, image_id in enumerate(all_image_ids):
        mask = np.zeros(IMG_SHAPE, dtype=np.uint8)
        if image_id in grouped.groups:
            rows = grouped.get_group(image_id)
            for _, row in rows.iterrows():
                class_mask = rle_decode(row.EncodedPixels, shape=IMG_SHAPE)
                mask[class_mask == 1] = row.ClassId

        out_name = os.path.splitext(image_id)[0] + ".png"
        cv2.imwrite(f"{OUT_DIR}/{out_name}", mask)

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(all_image_ids)}")

    print(f"Done. Masks saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
