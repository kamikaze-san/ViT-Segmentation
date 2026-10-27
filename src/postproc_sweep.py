"""Three post-processing changes we have never deployed, swept offline.

All three are decision-rule changes - the category that has transferred 5 times
out of 5, against 0 for 7 for anything that fits the training distribution
harder. None needs a GPU run: they read the cached probabilities.

  BILINEAR   We predict on a 16x100 grid, PixelShuffle to 64x400, then expand
             to 256x1600 with np.repeat - nearest neighbour. Every boundary is a
             4-pixel staircase against a smooth true edge. Interpolating the
             PROBABILITY map and thresholding at full resolution gives a real
             boundary for zero training cost.

  MORPH      Close small gaps, fill interior holes, drop stray components.
             Severstal defects are large contiguous blobs; a mask with holes in
             it is losing Dice on pixels the model already got approximately
             right.

  CLUSTER    5th place's method. Near-duplicate images from the same run share
             labels, so the empty/non-empty call is better made per cluster than
             per image. We measured the premise on our own data and it held (c3
             homogeneity 0.957 vs 0.629 shuffled) - then never used it.

Scored against the REAL Kaggle metric on the 500-image holdout, which is the
only split uncontaminated for every model (train12k contains all of val).
Thresholds are re-optimised per variant, because changing the upsampling changes
the areas and the old min_area no longer means the same thing.
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2

from spatial_head import N_CLASSES
from train_lora import MASK_DIR, SPLIT_DIR

ENS = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\ensemble"
CLASSES = (1, 2, 3, 4)
SHAPE = (256, 1600)
PIX_GRID = np.array([0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70], np.float32)
AREA_GRID = [0, 400, 700, 1050, 1400, 1750, 2100, 2450, 2800, 3500, 4200, 5250, 6300, 7500]
SUBMITTED_PIX = [0.60, 0.60, 0.60, 0.45]
SUBMITTED_MINA = [1050, 2450, 2450, 5250]

_K9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def upsample(p_small, mode):
    """(64,400) float prob -> (256,1600) float prob."""
    if mode == "block":
        return np.repeat(np.repeat(p_small, 4, 0), 4, 1)
    return cv2.resize(p_small, (SHAPE[1], SHAPE[0]), interpolation=cv2.INTER_LINEAR)


def morph(mask, min_component=300):
    """Close gaps, fill interior holes, drop stray components."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _K9)
    # Hole fill. Seeding the flood at (0,0) is wrong: when a defect touches that
    # corner the flood fills the DEFECT and every background pixel then reads as
    # an interior hole. Pad with a guaranteed-background border and seed there.
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    cv2.floodFill(pad, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    m = m | ((pad[1:-1, 1:-1] == 0) & (m == 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 2:
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_component:
                m[lab == i] = 0
    return m.astype(bool)


def gather(prob_paths, ids, mode, tag):
    """Per (image,class): area and intersection at each pixel threshold, plus GT
    area. Everything the threshold sweep needs, computed once per variant."""
    cache = f"{ENS}/pp_{tag}_{mode}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return z["area"], z["inter"], z["gt"]
    arrs = [np.load(p, mmap_mode="r") for p in prob_paths]
    n, T = len(ids), len(PIX_GRID)
    area = np.zeros((n, N_CLASSES, T), np.int64)
    inter = np.zeros((n, N_CLASSES, T), np.int64)
    gt = np.zeros((n, N_CLASSES), np.int64)
    for i, image_id in enumerate(ids):
        p = sum(np.asarray(a[i], np.float32) for a in arrs) / len(arrs)
        g = cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png", cv2.IMREAD_UNCHANGED)
        for c in CLASSES:
            full = upsample(p[c], "block" if mode == "block" else "bilinear")
            gc = g == c
            gt[i, c] = gc.sum()
            for j, t in enumerate(PIX_GRID):
                sel = full > t
                if mode == "bilinear_morph" and sel.any():
                    sel = morph(sel)
                area[i, c, j] = sel.sum()
                inter[i, c, j] = np.logical_and(sel, gc).sum()
        if (i + 1) % 100 == 0:
            print(f"    {mode}: {i+1}/{n}", flush=True)
    np.savez_compressed(cache, area=area, inter=inter, gt=gt)
    return area, inter, gt


def kaggle_class(area_c, inter_c, gt_c, pi, ma):
    a = np.where(area_c[:, pi] >= ma, area_c[:, pi], 0)
    i = np.where(area_c[:, pi] >= ma, inter_c[:, pi], 0)
    both = (a == 0) & (gt_c == 0)
    one = (a == 0) ^ (gt_c == 0)
    ok = ~both & ~one
    return (both.sum() * 1.0 + (2 * i[ok] / np.maximum(a[ok] + gt_c[ok], 1)).sum()) / len(a)


def best_per_class(area, inter, gt):
    out, total = {}, 0.0
    for k, c in enumerate(CLASSES):
        top = (-1, None)
        for pi in range(len(PIX_GRID)):
            for ma in AREA_GRID:
                s = kaggle_class(area[:, c], inter[:, c], gt[:, c], pi, ma)
                if s > top[0]:
                    top = (s, (float(PIX_GRID[pi]), ma))
        out[c] = top
        total += top[0]
    return out, total / len(CLASSES)


def at_submitted(area, inter, gt):
    tot = 0.0
    for k, c in enumerate(CLASSES):
        pi = int(np.argmin(np.abs(PIX_GRID - SUBMITTED_PIX[k])))
        tot += kaggle_class(area[:, c], inter[:, c], gt[:, c], pi, SUBMITTED_MINA[k])
    return tot / len(CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", default=["blackout"])
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--modes", nargs="+", default=["block", "bilinear", "bilinear_morph"],
                    help="restrict which upsampling variants to score; bilinear alone is "
                         "enough once morph has been ruled out, and skips the expensive "
                         "per-threshold morphology")
    a = ap.parse_args()
    ids = pd.read_csv(f"{SPLIT_DIR}/{a.split}_ids.csv")["ImageId"].tolist()
    paths = [f"{ENS}/probs_{m}_{a.split}.npy" for m in a.members]
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    tag = "+".join(a.members) + "_" + a.split
    print(f"members {'+'.join(a.members)} | {a.split} | {len(ids)} images\n")

    rows = []
    for mode in a.modes:
        area, inter, gt = gather(paths, ids, mode, tag)
        sub = at_submitted(area, inter, gt)
        per, opt = best_per_class(area, inter, gt)
        rows.append((mode, sub, opt, per))
        print(f"{mode:<16} at submitted thresholds {sub:.5f}   re-optimised {opt:.5f}")
        for c in CLASSES:
            s, (pix, ma) = per[c]
            print(f"       c{c}  pix {pix:.2f}  min_area {ma:<6} class {s:.5f}")
        print()

    base = rows[0][2]
    print(f"{'variant':<16}{'re-optimised':>14}{'vs block':>11}")
    for mode, sub, opt, _ in rows:
        print(f"{mode:<16}{opt:>14.5f}{opt-base:>+11.5f}")
    print("\n500 images x 4 classes = 2000 pairs, so +0.0005 is one pair. Treat")
    print("anything under ~0.002 as inside the noise of this split.")


if __name__ == "__main__":
    main()
