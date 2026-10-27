"""Sweep the class-weight exponent for the lightweight head.

weight_power controls how hard we correct class imbalance:
    0.0 = unweighted   (A1a's failure mode: predicts nothing for rare classes)
    1.0 = fully balanced inverse frequency (over-predicts, wrecks precision)

Trains one head per exponent off the cached features (seconds each), then
makes a SINGLE pass over the val split, applying every head to each image's
features so DINOv3 inference is shared instead of repeated per head.

Selection happens on val only. The test split is not touched here.
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn

from dino_features import load_model, extract_patch_features, PATCH_SIZE, DEFAULT_LAYERS
from train_head import build_head

CACHE_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\feature_cache"
DATA_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
MASK_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\masks"
SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints"

POWERS = [-0.7, -0.6, -0.5, -0.45, -0.4, -0.35]


def train_head(X, y, power, device, kind="linear", epochs=30, lr=1e-3, batch=8192):
    """X stays on CPU (multi-layer features are ~12 GB as float32, far past
    8 GB VRAM); batches are transferred per step instead."""
    counts = np.array([(y == c).sum() for c in range(5)], dtype=np.float64)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    w = w / w.mean()

    head = build_head(kind, dim=X.shape[1]).to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    yt_all = torch.from_numpy(y)
    n = len(y)
    for _ in range(epochs):
        head.train()
        idx = torch.randperm(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch].numpy()
            xb = torch.from_numpy(X[b]).float().to(device, non_blocking=True)
            yb = yt_all[b].to(device, non_blocking=True)
            opt.zero_grad()
            crit(head(xb), yb).backward()
            opt.step()
        sched.step()
    head.eval()
    torch.cuda.empty_cache()
    return head, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--head", default="linear", choices=["linear", "mlp"])
    p.add_argument("--multilayer", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tag = "_ml" if args.multilayer else ""
    layers = DEFAULT_LAYERS if args.multilayer else None
    X = np.load(f"{CACHE_DIR}/train{tag}_feats.npy")  # keep fp16 on CPU
    y = np.load(f"{CACHE_DIR}/train{tag}_labels.npy").astype(np.int64)
    print(f"Cache: {X.shape[0]} patches, dim {X.shape[1]} "
          f"({'multi-layer ' + str(DEFAULT_LAYERS) if args.multilayer else 'final layer'})\n")

    feat_dim = X.shape[1]
    heads = {}
    for pw in POWERS:
        h, w = train_head(X, y, pw, device, kind=args.head)
        heads[pw] = h
        print(f"trained weight_power={pw:.2f}  weights={np.round(w, 3).tolist()}")
    del X, y

    model, norm, _ = load_model(device)
    ids = pd.read_csv(f"{SPLIT_DIR}/{args.split}_ids.csv")["ImageId"].tolist()
    if args.n:
        ids = ids[:args.n]
    print(f"\nEvaluating {len(ids)} '{args.split}' images (one DINOv3 pass, all heads)\n")

    stats = {pw: {"i": np.zeros(5, np.int64), "p": np.zeros(5, np.int64), "g": np.zeros(5, np.int64)}
             for pw in POWERS}

    for n, image_id in enumerate(ids):
        image_bgr = cv2.imread(f"{DATA_DIR}/train_images/{image_id}")
        gt = cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png", cv2.IMREAD_UNCHANGED)
        tokens, grid = extract_patch_features(model, norm, device, image_bgr, layers=layers)
        t = torch.from_numpy(tokens).float().to(device)

        for pw, head in heads.items():
            with torch.no_grad():
                pp = head(t).argmax(1).cpu().numpy().reshape(grid).astype(np.uint8)
            pred = np.repeat(np.repeat(pp, PATCH_SIZE, 0), PATCH_SIZE, 1)
            s = stats[pw]
            for c in range(5):
                pc, gc = pred == c, gt == c
                s["i"][c] += np.logical_and(pc, gc).sum()
                s["p"][c] += pc.sum()
                s["g"][c] += gc.sum()

        if (n + 1) % 300 == 0:
            print(f"  {n + 1}/{len(ids)}")

    print(f"\n{'power':>7} {'bg':>7} {'c1':>7} {'c2':>7} {'c3':>7} {'c4':>7} {'mean1-4':>9}")
    best = (-1, None)
    for pw in POWERS:
        s = stats[pw]
        d = [(2 * s["i"][c] / (s["p"][c] + s["g"][c])) if (s["p"][c] + s["g"][c]) > 0 else np.nan
             for c in range(5)]
        m = np.nanmean(d[1:])
        print(f"{pw:>7.2f} {d[0]:>7.4f} {d[1]:>7.4f} {d[2]:>7.4f} {d[3]:>7.4f} {d[4]:>7.4f} {m:>9.4f}")
        if m > best[0]:
            best = (m, pw)

    print(f"\nBest on {args.split}: weight_power={best[1]} (mean Dice c1-4 = {best[0]:.4f})")
    path = f"{OUT_DIR}/head_{args.head}{tag}_best_wp{best[1]}.pt"
    torch.save({"state_dict": heads[best[1]].state_dict(), "kind": args.head,
                "dim": feat_dim, "layers": layers}, path)
    print(f"Saved best head to {path}")


if __name__ == "__main__":
    main()
