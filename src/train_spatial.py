"""Train + evaluate spatial heads over the memmapped DINOv3 feature cache.

Reads one image's feature map at a time (~4.9 MB, contiguous) so the 43 GB
train cache is never loaded into RAM.

Increments (each compared on VAL; test is touched once at the very end):
  A  --head mlp  --upscale 1              per-patch baseline on this pipeline
  A' --head conv --upscale 1              + spatial context
  B  --head conv --upscale 4              + sub-patch resolution
  C  --head conv --upscale 4 --dice 1.0   + Dice loss term
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn

from spatial_head import build, patch_labels_at_stride, soft_dice_loss, GRID, N_CLASSES

CACHE = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\spatial_cache"
MASK_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\masks"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints"
CLASS_NAMES = ["background", "defect_1", "defect_2", "defect_3", "defect_4"]


def load_split(split, stride):
    feats = np.load(f"{CACHE}/{split}_spatial.npy", mmap_mode="r")
    ids = pd.read_csv(f"{CACHE}/{split}_order.csv")["ImageId"].tolist()
    gh, gw = 256 // stride, 1600 // stride
    labels = np.zeros((len(ids), gh, gw), dtype=np.uint8)
    for i, image_id in enumerate(ids):
        m = cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png", cv2.IMREAD_UNCHANGED)
        labels[i] = patch_labels_at_stride(m, stride)
    return feats, labels, ids


def batches(n, bs, shuffle, rng=None):
    idx = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield np.sort(idx[i:i + bs])  # sorted -> more sequential memmap reads


def to_gpu(feats, sel, device):
    x = torch.from_numpy(np.ascontiguousarray(feats[sel])).float()
    x = x.permute(0, 2, 1).reshape(len(sel), -1, GRID[0], GRID[1])
    return x.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(head, feats, labels, device, bs=8, stride=16):
    head.eval()
    inter = np.zeros(N_CLASSES, np.int64)
    psum = np.zeros(N_CLASSES, np.int64)
    gsum = np.zeros(N_CLASSES, np.int64)
    for sel in batches(len(labels), bs, shuffle=False):
        x = to_gpu(feats, sel, device)
        pred = head(x).argmax(1).cpu().numpy()
        gt = labels[sel]
        for c in range(N_CLASSES):
            pc, gc = pred == c, gt == c
            inter[c] += np.logical_and(pc, gc).sum()
            psum[c] += pc.sum()
            gsum[c] += gc.sum()
    dice = [(2 * inter[c] / (psum[c] + gsum[c])) if (psum[c] + gsum[c]) > 0 else np.nan
            for c in range(N_CLASSES)]
    return np.array(dice)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="conv", choices=["mlp", "mlp_deep", "conv"])
    p.add_argument("--upscale", type=int, default=1)
    p.add_argument("--weight-power", type=float, default=-0.5)
    p.add_argument("--dice", type=float, default=0.0, help="weight on soft-Dice term")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--tag", default="")
    p.add_argument("--max-train", type=int, default=None,
                   help="use only the first N train images (for cheap hyperparameter calibration)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--hidden", type=int, default=256,
                   help="head width - the capacity knob for the scaling ablation")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stride = 16 // args.upscale
    print(f"head={args.head} upscale={args.upscale} -> output stride {stride}px, "
          f"wp={args.weight_power} dice={args.dice}")

    tr_f, tr_y, _ = load_split("train", stride)
    va_f, va_y, _ = load_split("val", stride)
    if args.max_train:
        tr_f, tr_y = tr_f[:args.max_train], tr_y[:args.max_train]
    print(f"train {tr_f.shape}, val {va_f.shape}, labels at {tr_y.shape[1:]}")

    counts = np.array([(tr_y == c).sum() for c in range(N_CLASSES)], dtype=np.float64)
    w = (counts.sum() / np.maximum(counts, 1)) ** args.weight_power
    w = w / w.mean()
    print("class weights:", np.round(w, 3).tolist())

    head = build(args.head, dim=tr_f.shape[2], upscale=args.upscale, hidden=args.hidden).to(device)
    print(f"trainable params: {sum(q.numel() for q in head.parameters()):,}")

    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(0)

    best = (-1, None)
    for ep in range(args.epochs):
        head.train()
        tot, nb = 0.0, 0
        for sel in batches(len(tr_y), args.bs, shuffle=True, rng=rng):
            x = to_gpu(tr_f, sel, device)
            y = torch.from_numpy(tr_y[sel]).long().to(device)
            opt.zero_grad()
            logits = head(x)
            loss = crit(logits, y)
            if args.dice > 0:
                loss = loss + args.dice * soft_dice_loss(logits, y)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        d = evaluate(head, va_f, va_y, device, bs=args.bs, stride=stride)
        m = np.nanmean(d[1:])
        flag = ""
        if m > best[0]:
            best = (m, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
            flag = "  <- best"
        print(f"ep {ep+1:2d} | loss {tot/nb:.4f} | val "
              f"c1 {d[1]:.4f} c2 {d[2]:.4f} c3 {d[3]:.4f} c4 {d[4]:.4f} | mean {m:.4f}{flag}")

    name = f"spatial_{args.head}_up{args.upscale}_h{args.hidden}_wp{args.weight_power}_dice{args.dice}{args.tag}"
    path = f"{OUT_DIR}/{name}.pt"
    torch.save({"state_dict": best[1], "kind": args.head, "upscale": args.upscale,
                "dim": tr_f.shape[2], "stride": stride, "hidden": args.hidden,
                "val_mean_dice": best[0]}, path)
    print(f"\nBEST val mean Dice (c1-4): {best[0]:.4f}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
