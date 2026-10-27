"""Kaggle Severstal metric for a full fine-tuned DINOv3 checkpoint.

kaggle_metric.py reads the frozen-feature spatial cache, so it cannot score a
model whose backbone has been fine-tuned. This runs the backbone live and
computes the identical two metrics:

  OUR METRIC     global-accumulation Dice per class, averaged over classes 1-4.
  KAGGLE METRIC  mean Dice over every (ImageId, ClassId) pair, with empty
                 prediction on empty ground truth scoring 1.0. Most pairs are
                 empty-empty, so this is dominated by correctly predicting
                 nothing and is NOT comparable to our mean-over-defect-classes.

The model runs ONCE and per-(image,class) areas are cached, so a min-area sweep
costs nothing extra. Note min_area thresholds TOTAL predicted area for a class
in an image, not connected components - matching kaggle_metric.py's actual
implementation (whose --help text said "components" but did the same thing).
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

from spatial_head import build, N_CLASSES
from train_lora import MODEL_ID, DATA_DIR, MASK_DIR, SPLIT_DIR, CKPT_DIR, GRID

CLASS_NAMES = ["background", "defect_1", "defect_2", "defect_3", "defect_4"]


@torch.no_grad()
def collect(ckpt_name, split, device, bs=4):
    """One pass over the split. Returns per-(image,class) predicted area,
    ground-truth area and intersection, for classes 1-4."""
    ck = torch.load(f"{CKPT_DIR}/{ckpt_name}", map_location=device, weights_only=False)
    taps = tuple(ck["tap_layers"])
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    model.load_state_dict(ck["backbone"])
    head = build("conv", dim=384 * len(taps), upscale=4, hidden=ck["hidden"]).to(device).eval()
    head.load_state_dict(ck["head"])

    mean = np.array(proc.image_mean, dtype=np.float32)
    std = np.array(proc.image_std, dtype=np.float32)
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)
    ids = pd.read_csv(f"{SPLIT_DIR}/{split}_ids.csv")["ImageId"].tolist()

    P = np.zeros((len(ids), N_CLASSES), np.int64)   # predicted area
    G = np.zeros((len(ids), N_CLASSES), np.int64)   # ground-truth area
    I = np.zeros((len(ids), N_CLASSES), np.int64)   # intersection

    for i0 in range(0, len(ids), bs):
        sel = ids[i0:i0 + bs]
        imgs, masks = [], []
        for image_id in sel:
            img = cv2.imread(f"{DATA_DIR}/train_images/{image_id}")
            rgb = img[:, :, ::-1].astype(np.float32) / 255.0
            imgs.append(((rgb - mean) / std).transpose(2, 0, 1))
            masks.append(cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png",
                                    cv2.IMREAD_UNCHANGED))
        x = torch.from_numpy(np.stack(imgs)).to(device)
        with torch.autocast("cuda", enabled=False):
            out = model(pixel_values=x, output_hidden_states=True)
            parts = []
            for i in taps:
                h = out.hidden_states[i + 1][:, n_special:, :]
                parts.append(F.layer_norm(h, (h.shape[-1],)))
            tok = torch.cat(parts, dim=-1)
            feat = tok.permute(0, 2, 1).reshape(len(sel), -1, GRID[0], GRID[1])
            cells = head(feat).float().argmax(1)
        cells = cells.cpu().numpy().astype(np.uint8)

        for k in range(len(sel)):
            pred = np.repeat(np.repeat(cells[k], 4, 0), 4, 1)
            gt = masks[k]
            for c in range(N_CLASSES):
                pc, gc = pred == c, gt == c
                P[i0 + k, c] = pc.sum(); G[i0 + k, c] = gc.sum()
                I[i0 + k, c] = np.logical_and(pc, gc).sum()
        if (i0 + bs) % 400 < bs:
            print(f"  {min(i0 + bs, len(ids))}/{len(ids)}", flush=True)
    return P, G, I


def score(P, G, I, min_area=0):
    """Both metrics at a given min-area threshold, from cached areas."""
    Pm, Im = P.copy(), I.copy()
    if min_area:
        drop = Pm[:, 1:] < min_area
        Pm[:, 1:][drop] = 0
        Im[:, 1:][drop] = 0

    dice = np.array([(2 * Im[:, c].sum() / (Pm[:, c].sum() + G[:, c].sum()))
                     if (Pm[:, c].sum() + G[:, c].sum()) > 0 else np.nan
                     for c in range(N_CLASSES)])

    pair, n_empty, n_ok_empty = [], 0, 0
    for c in range(1, N_CLASSES):
        p, g, i = Pm[:, c], G[:, c], Im[:, c]
        both_empty = (p == 0) & (g == 0)
        one_empty = ((p == 0) ^ (g == 0))
        ok = ~both_empty & ~one_empty
        pair.append(np.ones(both_empty.sum()))
        pair.append(np.zeros(one_empty.sum()))
        pair.append(2 * i[ok] / np.maximum(p[ok] + g[ok], 1))
        n_empty += int((g == 0).sum())
        n_ok_empty += int(both_empty.sum())
    pair = np.concatenate(pair)
    return dice, pair, n_empty, n_ok_empty


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--min-areas", type=int, nargs="+", default=[0],
                   help="sweep values; choose on VAL, then apply the winner to test")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model: {a.ckpt}   split: {a.split}\n")
    P, G, I = collect(a.ckpt, a.split, device)

    print(f"\n{'min_area':>9}{'our mean 1-4':>15}{'KAGGLE':>10}{'empty-GT right':>17}")
    best = None
    for ma in a.min_areas:
        dice, pair, n_empty, n_ok = score(P, G, I, ma)
        m = np.nanmean(dice[1:])
        print(f"{ma:>9}{m:>15.4f}{pair.mean():>10.4f}{100*n_ok/max(n_empty,1):>16.1f}%")
        if best is None or pair.mean() > best[1]:
            best = (ma, pair.mean(), dice, pair, n_empty, n_ok)

    ma, kag, dice, pair, n_empty, n_ok = best
    print(f"\n--- detail at min_area={ma} ---")
    print("OUR METRIC (global-accumulation Dice per class):")
    for c in range(N_CLASSES):
        print(f"  {CLASS_NAMES[c]:>12}: {dice[c]:.4f}")
    print(f"  {'mean 1-4':>12}: {np.nanmean(dice[1:]):.4f}")
    print(f"\nKAGGLE METRIC (mean Dice over all (image,class) pairs, empty-empty = 1.0):")
    print(f"  pairs evaluated      : {len(pair)}")
    print(f"  empty-GT pairs       : {n_empty} ({100*n_empty/len(pair):.1f}%)")
    print(f"  of those, we got     : {n_ok} right ({100*n_ok/max(n_empty,1):.1f}%)")
    print(f"  KAGGLE-STYLE SCORE   : {kag:.4f}")
    print(f"\n  (published FPN+SE-ResNeXt50 private LB = 0.90038 - different test set, not head-to-head)")


if __name__ == "__main__":
    main()
