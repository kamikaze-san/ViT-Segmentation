"""Compute the ACTUAL Kaggle Severstal metric alongside our internal one.

Kaggle: mean Dice over every (ImageId, ClassId) pair in the test set, with the
convention that an empty prediction on empty ground truth scores 1.0. Most
pairs are empty-empty, so that metric is dominated by correctly predicting
nothing - it is NOT comparable to our mean-Dice-over-defect-classes.

Reporting both makes the published 0.90038 baseline interpretable.
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2
import torch

from spatial_head import build

CACHE = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\spatial_cache"
MASK_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\masks"
CKPT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints"
GRID = (16, 100)


@torch.no_grad()
def run(ckpt_name, split, device, bs=8, min_area=0):
    ck = torch.load(f"{CKPT_DIR}/{ckpt_name}", map_location=device, weights_only=False)
    head = build(ck["kind"], dim=ck["dim"], upscale=ck["upscale"]).to(device).eval()
    head.load_state_dict(ck["state_dict"])
    up = 16 // ck["upscale"]

    feats = np.load(f"{CACHE}/{split}_spatial.npy", mmap_mode="r")
    ids = pd.read_csv(f"{CACHE}/{split}_order.csv")["ImageId"].tolist()

    # global accumulation (our internal metric)
    inter = np.zeros(5, np.int64); psum = np.zeros(5, np.int64); gsum = np.zeros(5, np.int64)
    # per-(image,class) Dice list (Kaggle metric)
    pair_scores = []
    n_empty_pairs = n_correct_empty = 0

    for i0 in range(0, len(ids), bs):
        sel = np.arange(i0, min(i0 + bs, len(ids)))
        x = torch.from_numpy(np.ascontiguousarray(feats[sel])).float()
        x = x.permute(0, 2, 1).reshape(len(sel), -1, GRID[0], GRID[1]).to(device)
        cells = head(x).argmax(1).cpu().numpy().astype(np.uint8)

        for k, idx in enumerate(sel):
            pred = np.repeat(np.repeat(cells[k], up, 0), up, 1)
            gt = cv2.imread(f"{MASK_DIR}/{os.path.splitext(ids[idx])[0]}.png", cv2.IMREAD_UNCHANGED)

            for c in range(5):
                pc, gc = pred == c, gt == c
                inter[c] += np.logical_and(pc, gc).sum()
                psum[c] += pc.sum(); gsum[c] += gc.sum()

            for c in [1, 2, 3, 4]:
                pc, gc = pred == c, gt == c
                if min_area and pc.sum() < min_area:
                    pc = np.zeros_like(pc)
                p, g = pc.sum(), gc.sum()
                if p == 0 and g == 0:
                    pair_scores.append(1.0); n_empty_pairs += 1; n_correct_empty += 1
                elif p == 0 or g == 0:
                    pair_scores.append(0.0)
                    if g == 0:
                        n_empty_pairs += 1
                else:
                    pair_scores.append(2 * np.logical_and(pc, gc).sum() / (p + g))

    dice = np.array([(2 * inter[c] / (psum[c] + gsum[c])) if (psum[c] + gsum[c]) > 0 else np.nan
                     for c in range(5)])
    return dice, np.array(pair_scores), n_empty_pairs, n_correct_empty


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="spatial_conv_up4_wp0.3_dice1.0_C.pt")
    p.add_argument("--split", default="test")
    p.add_argument("--min-area", type=int, default=0,
                   help="drop predicted components smaller than this many pixels")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dice, pairs, n_empty, n_ok_empty = run(a.ckpt, a.split, device, min_area=a.min_area)

    print(f"model: {a.ckpt}   split: {a.split}   min_area: {a.min_area}\n")
    names = ["background", "defect_1", "defect_2", "defect_3", "defect_4"]
    print("OUR METRIC (global-accumulation Dice per class):")
    for c in range(5):
        print(f"  {names[c]:>12}: {dice[c]:.4f}")
    print(f"  {'mean 1-4':>12}: {np.nanmean(dice[1:]):.4f}")

    print(f"\nKAGGLE METRIC (mean Dice over all (image,class) pairs, empty-empty = 1.0):")
    print(f"  pairs evaluated      : {len(pairs)}")
    print(f"  empty-GT pairs       : {n_empty} ({100*n_empty/len(pairs):.1f}%)")
    print(f"  of those, we got     : {n_ok_empty} right ({100*n_ok_empty/max(n_empty,1):.1f}%)")
    print(f"  KAGGLE-STYLE SCORE   : {pairs.mean():.4f}")

    nonempty = pairs[:]  # recompute split for context
    print(f"\n  (for reference, published FPN+SE-ResNeXt50 private LB = 0.90038 on this metric)")


if __name__ == "__main__":
    main()
