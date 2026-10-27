"""Calibrate a single similarity-difference threshold for the prototype
segmentation method, using only the train split (never touches val/test).
Patch-level Dice is used as the calibration objective - since every patch
upsamples to an equal-area block, patch-level Dice closely approximates
pixel-level Dice for this method.
"""
import numpy as np
import pandas as pd
import torch
import cv2
from transformers import AutoImageProcessor, AutoModel

from rle_utils import rle_decode
from zero_shot_probe import get_patch_features, DATA_DIR, MODEL_ID, IMG_SHAPE
from prototype_segmentation import TARGET_CLASS

SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"


def patch_label_from_mask(mask, grid_shape):
    gh, gw = grid_shape
    mask_small = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return (mask_small > 0).astype(np.uint8).reshape(-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    defect_proto = np.load("prototype_class2_defect.npy")
    bg_proto = np.load("prototype_class2_bg.npy")

    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    train_ids = set(pd.read_csv(f"{SPLIT_DIR}/train_ids.csv")["ImageId"])
    class_rows = df[(df.ClassId == TARGET_CLASS) & (df.ImageId.isin(train_ids))]

    all_scores, all_labels = [], []
    for i, (_, row) in enumerate(class_rows.iterrows()):
        img_path = f"{DATA_DIR}/train_images/{row.ImageId}"
        image_bgr = cv2.imread(img_path)
        mask = rle_decode(row.EncodedPixels, shape=IMG_SHAPE)

        patch_tokens, grid_shape = get_patch_features(model, processor, image_bgr, device)
        labels = patch_label_from_mask(mask, grid_shape)

        norm_tokens = patch_tokens / np.linalg.norm(patch_tokens, axis=1, keepdims=True)
        scores = norm_tokens @ defect_proto - norm_tokens @ bg_proto

        all_scores.append(scores)
        all_labels.append(labels)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(class_rows)}")

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    print(f"Total patches: {len(scores)}, positive: {labels.sum()}")

    best_dice, best_thresh = -1, None
    for thresh in np.linspace(scores.min(), scores.max(), 200):
        pred = (scores > thresh).astype(np.uint8)
        inter = np.logical_and(pred == 1, labels == 1).sum()
        denom = pred.sum() + labels.sum()
        dice = (2 * inter / denom) if denom > 0 else 0
        if dice > best_dice:
            best_dice, best_thresh = dice, thresh

    print(f"\nBest patch-level Dice on train calibration set: {best_dice:.4f} at threshold={best_thresh:.5f}")
    np.save("prototype_class2_threshold.npy", np.array([best_thresh]))
    print("Saved threshold.")


if __name__ == "__main__":
    main()
