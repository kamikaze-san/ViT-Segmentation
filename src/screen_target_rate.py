"""Label-free model screening on the COMPETITION test images.

Our local test split is drawn from train_images and has been shown to be
ANTI-correlated with hidden-test performance (Spearman -1.00 across B7/B8/B9).
So it cannot be used to rank models. What it turned out we can use is the
predicted-positive rate on the 5506 competition test images: across the three
submitted models it ordered perfectly (inversely) with the private score.

    model          local Dice   nonempty   private
    B7 unaug         0.7209       18.4%    0.85750
    B8 aug 5e-5      0.7161       16.9%    0.86798
    B9 aug 1e-4      0.7104       16.3%    0.87237

That rate needs no labels, so it can be computed here in ~5 minutes for any
checkpoint and used to decide whether a submission is worth spending.

Two caveats worth keeping in view:
  - n=3. This is the best signal we have and it is measured on the TARGET
    distribution, but it is not proven.
  - Lower is not unconditionally better. Training-set rate is 14.1%; pushing
    below that means suppressing real defects. The target is to approach it
    from above, not to minimise.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoImageProcessor

from spatial_head import build, N_CLASSES
from train_lora import MODEL_ID, DATA_DIR, CKPT_DIR, GRID

TRAIN_RATE = 0.141          # (image,class) positive rate in train.csv
REFERENCE = {"B7 unaug 1e-4": (18.4, 0.85750), "B8 aug 5e-5": (16.9, 0.86798),
             "B9 aug 1e-4": (16.3, 0.87237)}


def build_model(ck, device):
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    if "backbone" in ck:
        model.load_state_dict(ck["backbone"])
    elif "lora" in ck:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=ck["rank"], lora_alpha=ck["alpha"], lora_dropout=0.0,
                         bias="none", target_modules=["q_proj", "k_proj", "v_proj"])
        model = get_peft_model(model, cfg).to(device).eval()
        model.load_state_dict(ck["lora"], strict=False)
    taps = tuple(ck["tap_layers"])
    head = build("conv", dim=384 * len(taps), upscale=4, hidden=ck["hidden"]).to(device).eval()
    head.load_state_dict(ck["head"])
    return model, head, taps


@torch.no_grad()
def rate(ckpt_name, device, bs=4, limit=None):
    ck = torch.load(f"{CKPT_DIR}/{ckpt_name}", map_location=device, weights_only=False)
    model, head, taps = build_model(ck, device)
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    mean = np.array(proc.image_mean, np.float32)
    std = np.array(proc.image_std, np.float32)
    n_special = 1 + getattr(AutoConfig.from_pretrained(MODEL_ID), "num_register_tokens", 0)

    files = sorted(os.listdir(f"{DATA_DIR}/test_images"))
    if limit:
        files = files[:limit]
    pos = np.zeros(N_CLASSES, np.int64)

    for i0 in range(0, len(files), bs):
        chunk = files[i0:i0 + bs]
        x = np.stack([(((cv2.imread(f"{DATA_DIR}/test_images/{f}")[:, :, ::-1]
                         .astype(np.float32) / 255.0) - mean) / std).transpose(2, 0, 1)
                      for f in chunk])
        out = model(pixel_values=torch.from_numpy(x).to(device), output_hidden_states=True)
        parts = [F.layer_norm(out.hidden_states[t + 1][:, n_special:, :],
                              (out.hidden_states[t + 1].shape[-1],)) for t in taps]
        tok = torch.cat(parts, dim=-1)
        cells = head(tok.permute(0, 2, 1).reshape(len(chunk), -1, GRID[0], GRID[1])) \
            .float().argmax(1).cpu().numpy()
        for k in range(len(chunk)):
            for c in range(1, N_CLASSES):
                if (cells[k] == c).any():
                    pos[c] += 1
        if (i0 + bs) % 1000 < bs:
            print(f"    {min(i0+bs, len(files))}/{len(files)}", flush=True)
    del model, head
    torch.cuda.empty_cache()
    return pos, len(files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    device = "cuda"

    print(f"reference (submitted models):")
    for n, (r, s) in REFERENCE.items():
        print(f"    {n:<16}{r:5.1f}%   private {s:.5f}")
    print(f"    {'train.csv rate':<16}{TRAIN_RATE*100:5.1f}%   <- approaching this from above is the goal\n")

    for c in a.ckpts:
        print(f"### {c}")
        pos, n = rate(c, device, limit=a.limit)
        tot = pos[1:].sum()
        pct = 100 * tot / (n * 4)
        print(f"  images {n} | pairs {n*4} | non-empty {tot} ({pct:.1f}%)")
        print("  per class: " + "  ".join(
            f"c{i}={100*pos[i]/n:.1f}%" for i in range(1, N_CLASSES)))
        delta = pct - TRAIN_RATE * 100
        print(f"  excess over train rate: {delta:+.1f} pts", end="")
        best = min(REFERENCE.values(), key=lambda v: abs(v[0] - pct))
        print(f" | closest submitted: {best[0]:.1f}% -> {best[1]:.5f}\n")


if __name__ == "__main__":
    main()
