"""How much Dice is lost purely to patch-grid quantisation?

Takes ground-truth masks, collapses them to per-patch labels on the 16x100
grid (exactly as our head predicts), upsamples back to 256x1600 in blocks,
and scores against the original full-resolution mask.

This is the CEILING for any per-patch classifier at this grid resolution
with 100% perfect classification. No model involved.
"""
import os
import numpy as np
import pandas as pd
import cv2

from cache_features import patch_labels_multiclass
from dino_features import PATCH_SIZE

MASK_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\masks"
SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
CLASS_NAMES = {0: "background", 1: "defect_1", 2: "defect_2", 3: "defect_3", 4: "defect_4"}


def oracle_for_grid(ids, grid):
    n = 5
    inter = np.zeros(n, np.int64)
    psum = np.zeros(n, np.int64)
    gsum = np.zeros(n, np.int64)
    ph, pw = 256 // grid[0], 1600 // grid[1]

    for image_id in ids:
        gt = cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png", cv2.IMREAD_UNCHANGED)
        lab = patch_labels_multiclass_grid(gt, grid)
        pred = np.repeat(np.repeat(lab.reshape(grid), ph, 0), pw, 1)
        for c in range(n):
            pc, gc = pred == c, gt == c
            inter[c] += np.logical_and(pc, gc).sum()
            psum[c] += pc.sum()
            gsum[c] += gc.sum()

    dice = [(2 * inter[c] / (psum[c] + gsum[c])) if (psum[c] + gsum[c]) > 0 else np.nan
            for c in range(n)]
    return dice


def patch_labels_multiclass_grid(mask, grid):
    """Same rule as cache_features.patch_labels_multiclass but for any grid."""
    gh, gw = grid
    ph, pw = 256 // gh, 1600 // gw
    blocks = mask.reshape(gh, ph, gw, pw).transpose(0, 2, 1, 3).reshape(gh * gw, ph * pw)
    counts = np.stack([(blocks == c).sum(axis=1) for c in [1, 2, 3, 4]], axis=1)
    has = counts.sum(axis=1) > 0
    lab = np.zeros(gh * gw, dtype=np.uint8)
    lab[has] = counts[has].argmax(axis=1) + 1
    return lab


def main():
    ids = pd.read_csv(f"{SPLIT_DIR}/test_ids.csv")["ImageId"].tolist()
    print(f"Oracle ceiling on {len(ids)} test images (perfect per-patch classification)\n")

    print(f"{'grid':>10} {'stride':>7} {'bg':>7} {'c1':>7} {'c2':>7} {'c3':>7} {'c4':>7} {'mean1-4':>9}")
    for grid, name in [((16, 100), "16px"), ((32, 200), "8px"), ((64, 400), "4px"), ((128, 800), "2px")]:
        d = oracle_for_grid(ids, grid)
        m = np.nanmean(d[1:])
        print(f"{str(grid):>10} {name:>7} {d[0]:>7.4f} {d[1]:>7.4f} {d[2]:>7.4f} {d[3]:>7.4f} {d[4]:>7.4f} {m:>9.4f}")


if __name__ == "__main__":
    main()
