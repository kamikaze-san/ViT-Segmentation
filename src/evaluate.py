"""Shared Dice evaluation across all model variants (A1, A2, A3, B1, B2).

Computes per-class + mean Dice on a given split (default: held-out test),
using dataset-level (global) intersection/union accumulation rather than
per-image averaging - avoids degenerate 0/0 cases for rare classes.
"""
import argparse
import cv2
import numpy as np
import pandas as pd
import lightly_train

DATA_ROOT = r"C:\Users\NewGr\Downloads\ViT Project severstal"
RAW_IMAGES_DIR = f"{DATA_ROOT}\\severstal-steel-defect-detection\\train_images"
MASKS_DIR = f"{DATA_ROOT}\\severstal\\data\\processed\\masks"
SPLIT_IDS_DIR = f"{DATA_ROOT}\\severstal\\data\\processed"

CLASS_NAMES = {0: "background", 1: "defect_1", 2: "defect_2", 3: "defect_3", 4: "defect_4"}


def evaluate_model(model_path, split="test"):
    model = lightly_train.load_model(model_path)

    ids = pd.read_csv(f"{SPLIT_IDS_DIR}/{split}_ids.csv")["ImageId"].tolist()

    num_classes = len(CLASS_NAMES)
    intersection = np.zeros(num_classes, dtype=np.int64)
    pred_sum = np.zeros(num_classes, dtype=np.int64)
    gt_sum = np.zeros(num_classes, dtype=np.int64)

    for i, image_id in enumerate(ids):
        stem = image_id.rsplit(".", 1)[0]
        img_path = f"{RAW_IMAGES_DIR}/{image_id}"
        gt_path = f"{MASKS_DIR}/{stem}.png"

        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        pred = model.predict(img_path).cpu().numpy()

        for c in range(num_classes):
            pred_c = pred == c
            gt_c = gt == c
            intersection[c] += np.logical_and(pred_c, gt_c).sum()
            pred_sum[c] += pred_c.sum()
            gt_sum[c] += gt_c.sum()

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(ids)}")

    dice = np.zeros(num_classes)
    for c in range(num_classes):
        denom = pred_sum[c] + gt_sum[c]
        dice[c] = (2 * intersection[c] / denom) if denom > 0 else float("nan")

    print(f"\nDice per class ({split} split, {len(ids)} images):")
    for c in range(num_classes):
        print(f"  {CLASS_NAMES[c]:>12}: {dice[c]:.4f}")

    mean_dice_defects = np.nanmean(dice[1:])
    print(f"\n  Mean Dice (classes 1-4): {mean_dice_defects:.4f}")
    print(f"  Mean Dice (all classes incl. background): {np.nanmean(dice):.4f}")

    return dice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to exported .pt model")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    evaluate_model(args.model, args.split)
