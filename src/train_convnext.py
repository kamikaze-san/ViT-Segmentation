"""Train the ConvNeXt multi-scale head (spec A3: ViT vs ConvNeXt ablation).

Backbone is frozen and run on the fly (no feature cache - see convnext_head.py).
Reported val/test Dice is PIXEL-LEVEL: predictions are upsampled back to
256x1600 and scored with the same global accumulation as evaluate.py, so the
numbers are directly comparable to A1a-A1d and B1-B4.
"""
import argparse
import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn

from convnext_head import (load_backbone, preprocess, backbone_features,
                           ConvNeXtFPNHead, labels_at_stride, N_CLASSES)
from spatial_head import soft_dice_loss

DATA_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
MASK_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\masks"
SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints"
CLASS_NAMES = ["background", "defect_1", "defect_2", "defect_3", "defect_4"]


def precompute_labels(ids, stride, what="train"):
    """Cell labels for every image, computed once and held in RAM.

    At stride 4 this is 64*400 = 25.6 KB/image (~225 MB for the train split),
    and it removes the per-batch mask decode + block-majority computation that
    was ~40 of the ~65 ms per image in the training loop. It also gives EXACT
    class counts instead of a sampled estimate - the sampled version reported
    zero class-2 cells on small subsets and produced a degenerate class weight.
    """
    gh, gw = 256 // stride, 1600 // stride
    labs = np.zeros((len(ids), gh, gw), dtype=np.uint8)
    for i, image_id in enumerate(ids):
        m = cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png", cv2.IMREAD_UNCHANGED)
        labs[i] = labels_at_stride(m, stride)
    print(f"  precomputed {what} labels: {labs.shape} ({labs.nbytes/1e6:.0f} MB)")
    return labs


def read_images(ids, sel):
    return [cv2.imread(f"{DATA_DIR}/train_images/{ids[i]}") for i in sel]


def read_masks(ids, sel):
    return [cv2.imread(f"{MASK_DIR}/{os.path.splitext(ids[i])[0]}.png", cv2.IMREAD_UNCHANGED)
            for i in sel]


@torch.no_grad()
def evaluate(model, head, norm, device, ids, stride, bs=4, limit=None):
    """Pixel-level Dice: predictions upsampled to 256x1600."""
    head.eval()
    use = ids[:limit] if limit else ids
    inter = np.zeros(N_CLASSES, np.int64)
    psum = np.zeros(N_CLASSES, np.int64)
    gsum = np.zeros(N_CLASSES, np.int64)
    for i0 in range(0, len(use), bs):
        sel = list(range(i0, min(i0 + bs, len(use))))
        imgs, gts = read_images(use, sel), read_masks(use, sel)
        x = preprocess(imgs, norm, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(backbone_features(model, x))
        pred_cells = logits.float().argmax(1).cpu().numpy().astype(np.uint8)
        for k in range(len(imgs)):
            pred = np.repeat(np.repeat(pred_cells[k], stride, 0), stride, 1)
            gt = gts[k]
            for c in range(N_CLASSES):
                pc, gc = pred == c, gt == c
                inter[c] += np.logical_and(pc, gc).sum()
                psum[c] += pc.sum()
                gsum[c] += gc.sum()
    return np.array([(2 * inter[c] / (psum[c] + gsum[c])) if (psum[c] + gsum[c]) > 0 else np.nan
                     for c in range(N_CLASSES)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upscale", type=int, default=1, help="1 -> stride 4, 2 -> stride 2")
    p.add_argument("--weight-power", type=float, default=0.30)
    p.add_argument("--dice", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--proj", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    model, norm, device = load_backbone()
    head = ConvNeXtFPNHead(proj=args.proj, hidden=args.hidden, upscale=args.upscale).to(device)
    stride = head.output_stride
    print(f"ConvNeXt FPN head: upscale={args.upscale} -> output stride {stride}px, "
          f"params {sum(q.numel() for q in head.parameters()):,}")

    tr_ids = pd.read_csv(f"{SPLIT_DIR}/train_ids.csv")["ImageId"].tolist()
    va_ids = pd.read_csv(f"{SPLIT_DIR}/val_ids.csv")["ImageId"].tolist()
    if args.max_train:
        tr_ids = tr_ids[:args.max_train]
    print(f"train {len(tr_ids)}, val {len(va_ids)} (val_limit={args.val_limit})")

    tr_labs = precompute_labels(tr_ids, stride, "train")
    counts = np.array([(tr_labs == c).sum() for c in range(N_CLASSES)], dtype=np.float64)
    if (counts == 0).any():
        print(f"WARNING: classes with zero cells: {np.where(counts == 0)[0].tolist()} "
              f"- their weights are meaningless")
    w = (counts.sum() / np.maximum(counts, 1)) ** args.weight_power
    w = w / w.mean()
    print("cell counts:", counts.astype(int).tolist())
    print("class weights:", np.round(w, 3).tolist())

    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(0)

    best = (-1, None)
    for ep in range(args.epochs):
        head.train()
        order = rng.permutation(len(tr_ids))
        tot, nb, t0 = 0.0, 0, time.time()
        for i0 in range(0, len(order), args.bs):
            sel = order[i0:i0 + args.bs]
            imgs = read_images(tr_ids, sel)
            x = preprocess(imgs, norm, device)
            y = torch.from_numpy(tr_labs[sel]).long().to(device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(backbone_features(model, x))
                loss = crit(logits.float(), y)
                if args.dice > 0:
                    loss = loss + args.dice * soft_dice_loss(logits.float(), y)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        d = evaluate(model, head, norm, device, va_ids, stride, bs=args.bs, limit=args.val_limit)
        m = np.nanmean(d[1:])
        flag = ""
        if m > best[0]:
            best = (m, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
            flag = "  <- best"
        print(f"ep {ep+1:2d} | loss {tot/nb:.4f} | {time.time()-t0:5.0f}s | val "
              f"c1 {d[1]:.4f} c2 {d[2]:.4f} c3 {d[3]:.4f} c4 {d[4]:.4f} | mean {m:.4f}{flag}", flush=True)

    name = f"convnext_fpn_up{args.upscale}_wp{args.weight_power}_dice{args.dice}{args.tag}"
    path = f"{OUT_DIR}/{name}.pt"
    torch.save({"state_dict": best[1], "upscale": args.upscale, "stride": stride,
                "proj": args.proj, "hidden": args.hidden, "val_mean_dice": float(best[0])}, path)
    print(f"\nBEST val mean Dice (c1-4): {best[0]:.4f}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
