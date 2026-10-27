"""5th place's cluster averaging, scored end-to-end on the real Kaggle metric.

We already validated the PREMISE (cluster_test.py: c3 label homogeneity 0.957
against a 0.629 shuffled baseline) and then never used it. This scores the
actual method: borrow empty/non-empty evidence from an image's near-duplicates
before deciding.

    blended_maxp = alpha * own_maxp + (1 - alpha) * cluster_mean_maxp

alpha = 1 is our current per-image rule, so the baseline is inside the sweep
rather than quoted from elsewhere.

Measured on strong16 x val, NOT blackout x holdout, for two reasons:
  - strong16 trained on the 8.8k train split, so its val is genuinely unseen.
    blackout trained on train12k, which contains all of val.
  - clustering needs the duplicate groups intact. Our holdout is a random 500 of
    12,568, so a group of four near-duplicates typically has ONE member in it -
    clustering cannot recover a group that was split across the splits. 1,885
    images keep far more of each group together. This understates nothing; the
    real test set has all 5,506 images present at once, which is the friendliest
    case of all.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ENS = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\ensemble"
CLASSES = (1, 2, 3, 4)
AREA_GRID = [0, 200, 400, 600, 900, 1400, 2000, 3000, 4200]
LABEL_GRID = [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
ALPHAS = [1.0, 0.85, 0.70, 0.50, 0.30, 0.0]
KS = [200, 400, 600, 900]


def score(area_c, inter_c, gt_c, gate, pi, ma):
    keep = gate & (area_c[:, pi] >= ma)
    a = np.where(keep, area_c[:, pi], 0)
    i = np.where(keep, inter_c[:, pi], 0)
    both = (a == 0) & (gt_c == 0)
    one = (a == 0) ^ (gt_c == 0)
    ok = ~both & ~one
    return (both.sum() * 1.0 + (2 * i[ok] / np.maximum(a[ok] + gt_c[ok], 1)).sum()) / len(a)


def main():
    z = np.load(f"{ENS}/gathered_strong16_val.npz")
    maxp, area, inter, gt, pix = z["maxp"], z["area"], z["inter"], z["gt"], z["pix"]
    X = np.load(f"{ENS}/feat_val.npy")
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    n = len(X)
    print(f"strong16 x val | {n} images (clean: strong16 trained on the 8.8k split)\n")

    # per-class pix/min_area optimum with NO cluster gate, so the cluster
    # contribution is isolated rather than confounded with re-tuning
    base_cfg, base_total = {}, 0.0
    allow = np.ones(n, bool)
    for c in CLASSES:
        top = (-1, None)
        for pi in range(len(pix)):
            for ma in AREA_GRID:
                s = score(area[:, c], inter[:, c], gt[:, c], allow, pi, ma)
                if s > top[0]:
                    top = (s, (pi, ma))
        base_cfg[c] = top[1]
        base_total += top[0]
    base_total /= len(CLASSES)
    print(f"baseline, per-image decision, no label gate : {base_total:.5f}")
    print(f"  per class: " + "  ".join(
        f"c{c} pix {pix[base_cfg[c][0]]:.2f} area {base_cfg[c][1]}" for c in CLASSES) + "\n")

    print(f"{'k':>5}{'alpha':>8}{'kaggle':>10}{'vs baseline':>13}   per-class label_t")
    best = (base_total, None)
    for k in KS:
        lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit(X).labels_
        # cluster mean of max-probability, per class
        cmean = np.zeros_like(maxp)
        for u in np.unique(lab):
            sel = lab == u
            cmean[sel] = maxp[sel].mean(0)
        for alpha in ALPHAS:
            blended = alpha * maxp + (1 - alpha) * cmean
            total, lts = 0.0, []
            for c in CLASSES:
                pi, ma = base_cfg[c]
                top = (-1, None)
                for lt in LABEL_GRID:
                    s = score(area[:, c], inter[:, c], gt[:, c],
                              blended[:, c] >= lt, pi, ma)
                    if s > top[0]:
                        top = (s, lt)
                total += top[0]; lts.append(top[1])
            total /= len(CLASSES)
            flag = ""
            if total > best[0]:
                best = (total, (k, alpha, lts)); flag = "  <-"
            print(f"{k:>5}{alpha:>8.2f}{total:>10.5f}{total-base_total:>+13.5f}   {lts}{flag}")

    print(f"\nbest {best[0]:.5f}  ({best[1]})")
    print(f"gain over per-image baseline: {best[0]-base_total:+.5f}")
    print("\n1885 images x 4 classes = 7540 pairs; +0.00013 is one pair.")
    print("alpha=1.0 rows are the per-image rule WITH a label gate added, so any")
    print("gain there is the label threshold, not clustering. Clustering's own")
    print("contribution is the gap between the best alpha<1 row and the alpha=1 row.")


if __name__ == "__main__":
    main()
