"""Rate-match the ensemble on the competition test set BEFORE spending a submission.

Averaging model probabilities flattens the maps: a peak one model calls 0.85 and
the other calls 0.55 averages to 0.70. At a FIXED pixel threshold the ensemble's
predicted areas therefore shrink and its positive rate falls, even though its
decisions are better. The holdout threshold sweep already showed this - it asked
for min_area x0.8, which was compensation, not disagreement.

Our submitted operating point (11.2%) sits where the rate->score slope has
already collapsed (0.0066/pt -> 0.0019/pt between the last two submissions), so
drifting BELOW it is not free. This measures where the ensemble actually lands
and what min_area brings it back, using no labels and no submission.

Records, per (image,class), the predicted area at each pixel threshold - a 2 MB
table from which any min_area can be swept offline. Reports each model alone
through the identical code path, so reproducing blackout's known 11.2% validates
the measurement before the ensemble number is trusted.
"""
import argparse
import itertools
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoImageProcessor

from spatial_head import build, N_CLASSES
from train_lora import MODEL_ID, DATA_DIR, CKPT_DIR, GRID

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\ensemble"
CLASSES = (1, 2, 3, 4)
PIX_GRID = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60], np.float32)
CELL_PX = 16                      # each stride-4 cell covers 4x4 = 16 full-res pixels

MODELS = {
    "12k_ep16": "fullft_lr0.0001_g0.85_h1024_12k_ep16.pt",
    "blackout": "fullft_lr0.0001_g0.85_h1024_blackout_ep16.pt",
}
# the submitted operating point, for reference
SUBMITTED_PIX = [0.60, 0.60, 0.60, 0.45]
SUBMITTED_MINA = [1050, 2450, 2450, 5250]


def load(ckpt_name, device):
    ck = torch.load(f"{CKPT_DIR}/{ckpt_name}", map_location=device, weights_only=False)
    m = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    m.load_state_dict(ck["backbone"])
    taps = tuple(ck["tap_layers"])
    h = build("conv", dim=384 * len(taps), upscale=4, hidden=ck["hidden"]).to(device).eval()
    h.load_state_dict(ck["head"])
    return m, h, taps


@torch.no_grad()
def tta_probs(model, head, taps, ns, x):
    """3-view TTA (none / hflip / vflip), each un-flipped before averaging.
    Same three views as the submitted kernel."""
    acc = 0.0
    for view in ("none", "h", "v"):
        xv = x if view == "none" else (torch.flip(x, [3]) if view == "h" else torch.flip(x, [2]))
        o = model(pixel_values=xv, output_hidden_states=True)
        tok = torch.cat([F.layer_norm(o.hidden_states[t + 1][:, ns:, :],
                                      (o.hidden_states[t + 1].shape[-1],)) for t in taps], -1)
        p = head(tok.permute(0, 2, 1).reshape(len(x), -1, GRID[0], GRID[1])).float().softmax(1)
        if view == "h":
            p = torch.flip(p, [3])
        elif view == "v":
            p = torch.flip(p, [2])
        acc = acc + p
    return acc / 3.0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    device = "cuda"
    os.makedirs(OUT, exist_ok=True)

    files = sorted(os.listdir(f"{DATA_DIR}/test_images"))
    if a.limit:
        files = files[:a.limit]
    n = len(files)

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    mean = np.array(proc.image_mean, np.float32); std = np.array(proc.image_std, np.float32)
    ns = 1 + getattr(AutoConfig.from_pretrained(MODEL_ID), "num_register_tokens", 0)

    nets = {k: load(v, device) for k, v in MODELS.items()}
    keys = list(MODELS) + ["ENSEMBLE"]
    # area[key][image, class, pixel_threshold] in FULL-RES pixels
    area = {k: np.zeros((n, N_CLASSES, len(PIX_GRID)), np.int32) for k in keys}
    pg = torch.from_numpy(PIX_GRID).to(device)

    print(f"{n} competition test images | {len(MODELS)} models x 3 TTA views "
          f"= {len(MODELS)*3} forward passes", flush=True)
    for i0 in range(0, n, a.bs):
        ch = files[i0:i0 + a.bs]
        x = np.stack([(((cv2.imread(f"{DATA_DIR}/test_images/{f}")[:, :, ::-1]
                         .astype(np.float32) / 255.0) - mean) / std).transpose(2, 0, 1) for f in ch])
        x = torch.from_numpy(x).to(device)

        per = {}
        for k, (m, h, taps) in nets.items():
            per[k] = tta_probs(m, h, taps, ns, x)
        per["ENSEMBLE"] = sum(per[k] for k in MODELS) / len(MODELS)

        for k, p in per.items():
            # (B, C, H, W) vs (T,) -> counts of cells above each threshold
            cnt = (p[:, CLASSES, :, :, None] > pg).sum((2, 3))       # (B, 4, T)
            area[k][i0:i0 + len(ch), 1:, :] = cnt.cpu().numpy() * CELL_PX
        if (i0 + a.bs) % 800 < a.bs:
            print(f"    {min(i0+a.bs, n)}/{n}", flush=True)

    np.savez_compressed(f"{OUT}/testrate_areas.npz", pix=PIX_GRID, files=np.array(files),
                        **{f"area_{k}": area[k] for k in keys})
    print(f"\nsaved {OUT}/testrate_areas.npz\n")
    report(area, keys, n)


def rate_at(area_k, pix_idx, min_a):
    """Predicted-positive rate over all (image,class) pairs at one setting."""
    keep = np.stack([area_k[:, c, pix_idx[j]] >= min_a[j] for j, c in enumerate(CLASSES)], 1)
    return keep.mean(), keep.mean(0)


def report(area, keys, n):
    pi = [PIX_GRID.tolist().index(p) for p in SUBMITTED_PIX]
    print("AT THE SUBMITTED OPERATING POINT "
          f"pix {SUBMITTED_PIX}  min_area {SUBMITTED_MINA}")
    print(f"{'model':<12}{'rate':>8}   per-class c1/c2/c3/c4")
    for k in keys:
        r, pc = rate_at(area[k], pi, SUBMITTED_MINA)
        print(f"{k:<12}{100*r:>7.2f}%   " + " ".join(f"{100*v:5.2f}" for v in pc))
    print("\n  (blackout should reproduce ~11.2% - that is the validity check)\n")

    print("ENSEMBLE min_area scaling, to rate-match blackout's 11.2%")
    print(f"{'scale':>7}{'min_area':>32}{'rate':>9}")
    for s in (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50):
        ma = [int(round(v * s)) for v in SUBMITTED_MINA]
        r, _ = rate_at(area["ENSEMBLE"], pi, ma)
        print(f"{s:>7.2f}{str(ma):>32}{100*r:>8.2f}%")


if __name__ == "__main__":
    main()
