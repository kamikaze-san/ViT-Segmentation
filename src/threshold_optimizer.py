"""Triple thresholding: the decision stage every top-8 Severstal solution had
and we entirely lack.

We currently take a bare argmax over the 5-way softmax. A pixel becomes class c
only if it beats background AND all other classes, and there is no notion of
confidence anywhere in the pipeline. Every gold solution instead ran three
per-class stages:

    label threshold   max-pixel probability for the (image,class) decision.
                      This IS the classifier - 3rd, 5th, 7th and 8th place all
                      skipped training a separate one and thresholded the
                      segmentation's own max probability instead. 7th place
                      lists "training a classifier" under what did NOT work.
    pixel threshold   which pixels to keep, ~0.3-0.55 rather than argmax.
                      Lower than argmax implies, so it also RECOVERS misses.
    min area          drop the whole mask below a per-class pixel count.
                      1st: 600/600/900/2000.  5th: 547/715/1145/2959, derived
                      as the 2% quantile of defect sizes in the training data.

Scored against the REAL Kaggle metric on labelled val, so no proxy is involved
and none of the earlier "the rate proxy cannot see this" caveats apply.

Reads the probability caches written by ensemble_test.py, so thresholds can be
optimised jointly with the choice of ensemble members - which matters, because
averaged probabilities are better calibrated and want different thresholds than
a single model does.
"""
import argparse
import itertools
import os
import numpy as np
import pandas as pd
import cv2

from spatial_head import N_CLASSES
from train_lora import MASK_DIR, SPLIT_DIR

ENS = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\ensemble"
CLASSES = (1, 2, 3, 4)


def gather(prob_paths, ids, split):
    """One pass over the cached probabilities -> per (image,class):
    max probability, and the sorted defect-pixel counts at several pixel
    thresholds. Everything the sweep needs, without re-reading the arrays."""
    cache = f"{ENS}/gathered_{'_'.join(os.path.basename(p).split('_')[1] for p in prob_paths)}_{split}.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        return z["maxp"], z["area"], z["inter"], z["gt"], z["pix"]

    arrs = [np.load(p, mmap_mode="r") for p in prob_paths]
    pix_grid = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60], np.float32)
    n = len(ids)
    maxp = np.zeros((n, N_CLASSES), np.float32)
    area = np.zeros((n, N_CLASSES, len(pix_grid)), np.int64)
    inter = np.zeros((n, N_CLASSES, len(pix_grid)), np.int64)
    gt = np.zeros((n, N_CLASSES), np.int64)

    for i, image_id in enumerate(ids):
        p = sum(np.asarray(a[i], np.float32) for a in arrs) / len(arrs)
        full = np.repeat(np.repeat(p, 4, 1), 4, 2)          # (5, 256, 1600)
        m = cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png", cv2.IMREAD_UNCHANGED)
        for c in CLASSES:
            pc = full[c]
            maxp[i, c] = pc.max()
            gc = (m == c)
            gt[i, c] = gc.sum()
            for j, t in enumerate(pix_grid):
                sel = pc > t
                area[i, c, j] = sel.sum()
                inter[i, c, j] = np.logical_and(sel, gc).sum()
        if (i + 1) % 400 == 0:
            print(f"    gathered {i+1}/{n}", flush=True)
    np.savez_compressed(cache, maxp=maxp, area=area, inter=inter, gt=gt, pix=pix_grid)
    return maxp, area, inter, gt, pix_grid


def kaggle(maxp, area, inter, gt, pix, label_t, pix_i, min_a):
    """Kaggle metric under one threshold setting. label_t / pix_i / min_a are
    per-class, indexed 0..3 for classes 1..4."""
    n, total = len(maxp), 0.0
    for k, c in enumerate(CLASSES):
        keep = (maxp[:, c] >= label_t[k]) & (area[:, c, pix_i[k]] >= min_a[k])
        a = np.where(keep, area[:, c, pix_i[k]], 0)
        i = np.where(keep, inter[:, c, pix_i[k]], 0)
        g = gt[:, c]
        both_empty = (a == 0) & (g == 0)
        one_empty = (a == 0) ^ (g == 0)
        ok = ~both_empty & ~one_empty
        total += both_empty.sum() * 1.0 + (2 * i[ok] / np.maximum(a[ok] + g[ok], 1)).sum()
    return total / (n * len(CLASSES))


def per_class_best(maxp, area, inter, gt, pix, label_grid, area_grid):
    """The classes are INDEPENDENT under this metric - a pair's score depends
    only on its own class - so each class can be optimised separately instead of
    searching the full product space."""
    best = {}
    n = len(maxp)
    for k, c in enumerate(CLASSES):
        g = gt[:, c]
        top = (-1, None)
        for lt in label_grid:
            for pi in range(len(pix)):
                for ma in area_grid:
                    keep = (maxp[:, c] >= lt) & (area[:, c, pi] >= ma)
                    a = np.where(keep, area[:, c, pi], 0)
                    i = np.where(keep, inter[:, c, pi], 0)
                    both = (a == 0) & (g == 0)
                    one = (a == 0) ^ (g == 0)
                    ok = ~both & ~one
                    s = (both.sum() * 1.0 + (2 * i[ok] / np.maximum(a[ok] + g[ok], 1)).sum()) / n
                    if s > top[0]:
                        top = (s, (lt, float(pix[pi]), ma, pi))
        best[c] = top
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--members", nargs="+", default=["strong16"])
    a = ap.parse_args()
    ids = pd.read_csv(f"{SPLIT_DIR}/{a.split}_ids.csv")["ImageId"].tolist()
    paths = [f"{ENS}/probs_{m}_{a.split}.npy" for m in a.members]
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} - run ensemble_test.py first to cache probabilities")

    print(f"members: {'+'.join(a.members)}   split: {a.split}   {len(ids)} images")
    maxp, area, inter, gt, pix = gather(paths, ids, a.split)

    base = kaggle(maxp, area, inter, gt, pix, [0.0]*4, [pix.tolist().index(0.50)]*4, [0]*4)
    print(f"\nbaseline (no thresholds, pixel>0.5)          : {base:.5f}")

    label_grid = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    area_grid = [0, 100, 200, 400, 600, 900, 1400, 2000, 3000]
    best = per_class_best(maxp, area, inter, gt, pix, label_grid, area_grid)

    print(f"\n{'cls':>4}{'label_t':>9}{'pixel_t':>9}{'min_area':>10}{'class score':>13}"
          f"{'vs no-thresh':>14}")
    lt, pi, ma = [], [], []
    for k, c in enumerate(CLASSES):
        s, (l, p_, m_, idx) = best[c]
        g = gt[:, c]
        nothing = ((g == 0).sum() * 1.0) / len(maxp)
        print(f"{c:>4}{l:>9.2f}{p_:>9.2f}{m_:>10}{s:>13.5f}"
              f"{'   (silent: %.5f)' % nothing:>14}")
        lt.append(l); pi.append(idx); ma.append(m_)
    opt = kaggle(maxp, area, inter, gt, pix, lt, pi, ma)
    print(f"\nOPTIMISED kaggle metric                      : {opt:.5f}")
    print(f"gain over no-threshold baseline               : {opt-base:+.5f}")
    print(f"\nchosen: label {lt}  pixel {[float(pix[i]) for i in pi]}  min_area {ma}")


if __name__ == "__main__":
    main()
