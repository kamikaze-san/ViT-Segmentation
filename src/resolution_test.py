"""Does running inference at 2x input resolution beat the stride-4 grid?

We predict on a 16x100 patch grid, PixelShuffle to 64x400, then expand to
256x1600 - so the finest distinction the model can draw is a 4x4 block. Bilinear
upsampling softened that boundary but could not add information the grid never
had. Feeding 512x3200 gives a 32x200 grid and a 128x800 prediction: effective
stride 2, genuinely finer, no retraining.

This is an INFERENCE-ONLY change, the category that is 6 for 6 in this project,
against 1 for 9 for anything that retrains. The head is fully convolutional so
it applies unchanged to a larger grid; DINOv3 uses RoPE, which extrapolates
across resolutions far better than learned position embeddings.

The risk is real though: the backbone saw 256x1600 in training, and at 2x the
patches cover half the physical area, so texture statistics per patch change.
RoPE handles the geometry, not the scale shift. Scored on the real Kaggle metric
on the holdout against the identical 1x pipeline.
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoImageProcessor

from spatial_head import build, N_CLASSES
from train_lora import MODEL_ID, DATA_DIR, MASK_DIR, SPLIT_DIR, CKPT_DIR

CLASSES = (1, 2, 3, 4)
SHAPE = (256, 1600)
PIX_GRID = np.array([0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70], np.float32)
AREA_GRID = [0, 400, 700, 1050, 1400, 2100, 2450, 3500, 5250, 7350]
SUBMITTED_PIX = [0.60, 0.60, 0.60, 0.45]
SUBMITTED_MINA = [1050, 2450, 2450, 5250]


@torch.no_grad()
def run(ckpt_name, ids, scale, device, bs):
    ck = torch.load(f"{CKPT_DIR}/{ckpt_name}", map_location=device, weights_only=False)
    model_id = ck.get("model_id") or MODEL_ID
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    model.load_state_dict(ck["backbone"])
    taps = tuple(ck["tap_layers"])
    dim = model.config.hidden_size
    head = build("conv", dim=dim * len(taps), upscale=4, hidden=ck["hidden"]).to(device).eval()
    head.load_state_dict(ck["head"])
    proc = AutoImageProcessor.from_pretrained(model_id)
    mean = np.array(proc.image_mean, np.float32); std = np.array(proc.image_std, np.float32)
    ns = 1 + getattr(AutoConfig.from_pretrained(model_id), "num_register_tokens", 0)
    gh, gw = 16 * scale, 100 * scale

    n, T = len(ids), len(PIX_GRID)
    area = np.zeros((n, N_CLASSES, T), np.int64)
    inter = np.zeros((n, N_CLASSES, T), np.int64)
    gt = np.zeros((n, N_CLASSES), np.int64)

    for i0 in range(0, n, bs):
        ch = ids[i0:i0 + bs]
        imgs = []
        for f in ch:
            im = cv2.imread(f"{DATA_DIR}/train_images/{f}")[:, :, ::-1]
            if scale != 1:
                im = cv2.resize(im, (SHAPE[1] * scale, SHAPE[0] * scale),
                                interpolation=cv2.INTER_CUBIC)
            imgs.append((((im.astype(np.float32) / 255.0) - mean) / std).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(imgs)).to(device)
        o = model(pixel_values=x, output_hidden_states=True)
        tok = torch.cat([F.layer_norm(o.hidden_states[t + 1][:, ns:, :],
                                      (o.hidden_states[t + 1].shape[-1],)) for t in taps], -1)
        p = head(tok.permute(0, 2, 1).reshape(len(ch), -1, gh, gw)).float().softmax(1)
        # whatever the grid, compare at full 256x1600
        p = F.interpolate(p, size=SHAPE, mode="bilinear", align_corners=False).cpu().numpy()
        for k, image_id in enumerate(ch):
            g = cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png", cv2.IMREAD_UNCHANGED)
            for c in CLASSES:
                gc = g == c
                gt[i0 + k, c] = gc.sum()
                for j, t in enumerate(PIX_GRID):
                    sel = p[k, c] > t
                    area[i0 + k, c, j] = sel.sum()
                    inter[i0 + k, c, j] = np.logical_and(sel, gc).sum()
        if (i0 + bs) % 100 < bs:
            print(f"    scale {scale}x: {min(i0+bs, n)}/{n}", flush=True)
    del model, head; torch.cuda.empty_cache()
    return area, inter, gt


def kag(area_c, inter_c, gt_c, pi, ma):
    a = np.where(area_c[:, pi] >= ma, area_c[:, pi], 0)
    i = np.where(area_c[:, pi] >= ma, inter_c[:, pi], 0)
    both = (a == 0) & (gt_c == 0); one = (a == 0) ^ (gt_c == 0); ok = ~both & ~one
    return (both.sum() * 1.0 + (2 * i[ok] / np.maximum(a[ok] + gt_c[ok], 1)).sum()) / len(a)


def score(area, inter, gt, label):
    sub = sum(kag(area[:, c], inter[:, c], gt[:, c],
                  int(np.argmin(np.abs(PIX_GRID - SUBMITTED_PIX[k]))), SUBMITTED_MINA[k])
              for k, c in enumerate(CLASSES)) / 4
    tot, cfg = 0.0, []
    for k, c in enumerate(CLASSES):
        top = (-1, None)
        for pi in range(len(PIX_GRID)):
            for ma in AREA_GRID:
                s = kag(area[:, c], inter[:, c], gt[:, c], pi, ma)
                if s > top[0]:
                    top = (s, (float(PIX_GRID[pi]), ma))
        tot += top[0]; cfg.append(top[1])
    print(f"{label:<10} at submitted {sub:.5f}   re-optimised {tot/4:.5f}   {cfg}")
    return sub, tot / 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fullft_lr0.0001_g0.85_h1024_blackout_ep16.pt")
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bs", type=int, default=2)
    a = ap.parse_args()
    ids = pd.read_csv(f"{SPLIT_DIR}/{a.split}_ids.csv")["ImageId"].tolist()
    if a.limit:
        ids = ids[:a.limit]
    print(f"{a.ckpt} | {a.split} | {len(ids)} images\n")
    r = {}
    for sc in (1, 2):
        area, inter, gt = run(a.ckpt, ids, sc, "cuda", a.bs if sc == 1 else 1)
        r[sc] = score(area, inter, gt, f"{sc}x")
    print(f"\n2x vs 1x   at submitted {r[2][0]-r[1][0]:+.5f}   re-optimised {r[2][1]-r[1][1]:+.5f}")
    print(f"{len(ids)} x 4 pairs; {4/(len(ids)*4):.5f} per 4 pairs. Under ~0.002 is noise.")


if __name__ == "__main__":
    main()
