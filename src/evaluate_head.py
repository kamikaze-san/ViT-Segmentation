"""Evaluate a trained lightweight head on frozen DINOv3 features.

Runs live inference (extract features at native resolution -> head -> argmax
-> upsample patch grid to full 256x1600) and computes per-class Dice using
the SAME global accumulation as evaluate.py, so numbers are directly
comparable with the A1a-A1d trained-decoder results.
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2
import torch

from dino_features import load_model, extract_patch_features, PATCH_SIZE, IMG_SHAPE
from train_head import build_head

DATA_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
MASK_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\masks"
SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"

CLASS_NAMES = {0: "background", 1: "defect_1", 2: "defect_2", 3: "defect_3", 4: "defect_4"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-path", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=None)
    args = p.parse_args()

    model, norm, device = load_model()
    ckpt = torch.load(args.head_path, map_location=device)
    dim = ckpt.get("dim", 384)
    layers = ckpt.get("layers", None)
    head = build_head(ckpt["kind"], dim=dim).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    print(f"Loaded head '{ckpt['kind']}' (dim {dim}, layers "
          f"{layers if layers else 'final only'}) from {args.head_path}")

    ids = pd.read_csv(f"{SPLIT_DIR}/{args.split}_ids.csv")["ImageId"].tolist()
    if args.n:
        ids = ids[:args.n]

    n_cls = 5
    inter = np.zeros(n_cls, dtype=np.int64)
    psum = np.zeros(n_cls, dtype=np.int64)
    gsum = np.zeros(n_cls, dtype=np.int64)

    for i, image_id in enumerate(ids):
        image_bgr = cv2.imread(f"{DATA_DIR}/train_images/{image_id}")
        stem = os.path.splitext(image_id)[0]
        gt = cv2.imread(f"{MASK_DIR}/{stem}.png", cv2.IMREAD_UNCHANGED)

        tokens, grid = extract_patch_features(model, norm, device, image_bgr, layers=layers)
        with torch.no_grad():
            logits = head(torch.from_numpy(tokens).float().to(device))
            pred_patch = logits.argmax(1).cpu().numpy().reshape(grid).astype(np.uint8)

        pred = np.repeat(np.repeat(pred_patch, PATCH_SIZE, axis=0), PATCH_SIZE, axis=1)

        for c in range(n_cls):
            pc, gc = pred == c, gt == c
            inter[c] += np.logical_and(pc, gc).sum()
            psum[c] += pc.sum()
            gsum[c] += gc.sum()

        if (i + 1) % 300 == 0:
            print(f"  {i + 1}/{len(ids)}")

    print(f"\nDice per class ({args.split} split, {len(ids)} images):")
    dice = np.zeros(n_cls)
    for c in range(n_cls):
        d = psum[c] + gsum[c]
        dice[c] = (2 * inter[c] / d) if d > 0 else float("nan")
        print(f"  {CLASS_NAMES[c]:>12}: {dice[c]:.4f}")
    print(f"\n  Mean Dice (classes 1-4): {np.nanmean(dice[1:]):.4f}")
    print(f"  Mean Dice (all classes): {np.nanmean(dice):.4f}")


if __name__ == "__main__":
    main()
