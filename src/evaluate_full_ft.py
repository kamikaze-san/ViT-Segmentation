"""Evaluate a full fine-tuned DINOv3 + head checkpoint in fp32.

Mirrors evaluate_lora.py, but loads a complete backbone state_dict instead of
reconstructing a PEFT wrapper. Runs without autocast by default so the number
is directly comparable to every frozen result from evaluate_spatial.py.
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
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--bf16", action="store_true", help="use autocast (matches training-time eval)")
    args = p.parse_args()

    device = "cuda"
    ck = torch.load(f"{CKPT_DIR}/{args.ckpt}", map_location=device, weights_only=False)
    taps = tuple(ck["tap_layers"])

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    missing, unexpected = model.load_state_dict(ck["backbone"], strict=True)
    print(f"backbone loaded (missing={len(missing)}, unexpected={len(unexpected)})")

    head = build("conv", dim=384 * len(taps), upscale=4, hidden=ck["hidden"]).to(device).eval()
    head.load_state_dict(ck["head"])
    print(f"checkpoint val (bf16, from training): {ck['val_mean_dice']:.4f}")
    print(f"evaluating '{args.split}' in {'bf16' if args.bf16 else 'fp32'}\n")

    mean = np.array(proc.image_mean, dtype=np.float32)
    std = np.array(proc.image_std, dtype=np.float32)
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)
    ids = pd.read_csv(f"{SPLIT_DIR}/{args.split}_ids.csv")["ImageId"].tolist()

    inter = np.zeros(N_CLASSES, np.int64)
    psum = np.zeros(N_CLASSES, np.int64)
    gsum = np.zeros(N_CLASSES, np.int64)

    for i0 in range(0, len(ids), args.bs):
        sel = ids[i0:i0 + args.bs]
        imgs, masks = [], []
        for image_id in sel:
            img = cv2.imread(f"{DATA_DIR}/train_images/{image_id}")
            rgb = img[:, :, ::-1].astype(np.float32) / 255.0
            imgs.append(((rgb - mean) / std).transpose(2, 0, 1))
            masks.append(cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png",
                                    cv2.IMREAD_UNCHANGED))
        x = torch.from_numpy(np.stack(imgs)).to(device)

        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 \
            else torch.autocast("cuda", enabled=False)
        with ctx:
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
                inter[c] += np.logical_and(pc, gc).sum()
                psum[c] += pc.sum(); gsum[c] += gc.sum()

        if (i0 + args.bs) % 400 < args.bs:
            print(f"  {min(i0 + args.bs, len(ids))}/{len(ids)}", flush=True)

    dice = np.array([(2 * inter[c] / (psum[c] + gsum[c])) if (psum[c] + gsum[c]) > 0 else np.nan
                     for c in range(N_CLASSES)])
    print(f"\nDice per class ({args.split}):")
    for c in range(N_CLASSES):
        print(f"  {CLASS_NAMES[c]:>12}: {dice[c]:.4f}")
    print(f"\n  Mean Dice (classes 1-4): {np.nanmean(dice[1:]):.4f}")


if __name__ == "__main__":
    main()
