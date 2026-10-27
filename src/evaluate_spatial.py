"""Pixel-level evaluation of saved spatial heads.

train_spatial.py's in-loop evaluate() scores on the PATCH GRID, which ignores
upsampling error and is not comparable across different output strides (a
stride-4 head is scored on a 64x400 grid, a stride-16 head on 16x100). It is
fine for epoch selection but not for reporting.

This script upsamples every head's prediction back to full 256x1600 and scores
against the original mask, using the same global accumulation as evaluate.py -
so numbers here ARE directly comparable to A1a-A1d and B1-B3.
"""
import argparse
import glob
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
CLASS_NAMES = ["background", "defect_1", "defect_2", "defect_3", "defect_4"]


@torch.no_grad()
def eval_head(ckpt_path, split, device, bs=8, limit=None):
    # weights_only=False: these are our own checkpoints and contain a numpy
    # scalar (val_mean_dice) that the torch>=2.6 safe unpickler rejects.
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    head = build(ck["kind"], dim=ck["dim"], upscale=ck["upscale"],
                 hidden=ck.get("hidden", 256)).to(device)
    head.load_state_dict(ck["state_dict"])
    head.eval()

    feats = np.load(f"{CACHE}/{split}_spatial.npy", mmap_mode="r")
    ids = pd.read_csv(f"{CACHE}/{split}_order.csv")["ImageId"].tolist()
    if limit:
        ids = ids[:limit]

    up = 16 // ck["upscale"]  # pixels per predicted cell
    inter = np.zeros(5, np.int64); psum = np.zeros(5, np.int64); gsum = np.zeros(5, np.int64)

    for i0 in range(0, len(ids), bs):
        sel = np.arange(i0, min(i0 + bs, len(ids)))
        x = torch.from_numpy(np.ascontiguousarray(feats[sel])).float()
        x = x.permute(0, 2, 1).reshape(len(sel), -1, GRID[0], GRID[1]).to(device)
        pred_cells = head(x).argmax(1).cpu().numpy().astype(np.uint8)

        for k, idx in enumerate(sel):
            pred = np.repeat(np.repeat(pred_cells[k], up, 0), up, 1)
            gt = cv2.imread(f"{MASK_DIR}/{os.path.splitext(ids[idx])[0]}.png", cv2.IMREAD_UNCHANGED)
            assert pred.shape == gt.shape, (pred.shape, gt.shape)
            for c in range(5):
                pc, gc = pred == c, gt == c
                inter[c] += np.logical_and(pc, gc).sum()
                psum[c] += pc.sum()
                gsum[c] += gc.sum()

    dice = np.array([(2 * inter[c] / (psum[c] + gsum[c])) if (psum[c] + gsum[c]) > 0 else np.nan
                     for c in range(5)])
    return dice, ck


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val")
    p.add_argument("--pattern", default="spatial_*.pt")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, args.pattern)))
    if not paths:
        print("no checkpoints matched", args.pattern)
        return

    print(f"PIXEL-LEVEL Dice on '{args.split}' (predictions upsampled to 256x1600)\n")
    print(f"{'checkpoint':<46}{'stride':>7}{'bg':>8}{'c1':>8}{'c2':>8}{'c3':>8}{'c4':>8}{'mean1-4':>10}")
    rows = []
    for path in paths:
        d, ck = eval_head(path, args.split, device, limit=args.limit)
        m = np.nanmean(d[1:])
        rows.append((m, os.path.basename(path)))
        print(f"{os.path.basename(path):<46}{16//ck['upscale']:>7}"
              f"{d[0]:>8.4f}{d[1]:>8.4f}{d[2]:>8.4f}{d[3]:>8.4f}{d[4]:>8.4f}{m:>10.4f}")

    rows.sort(reverse=True)
    print(f"\nBest: {rows[0][1]}  mean Dice c1-4 = {rows[0][0]:.4f}")


if __name__ == "__main__":
    main()
