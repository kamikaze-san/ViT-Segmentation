"""Per-class test Dice AND the Kaggle metric, for every checkpoint family.

One evaluator for all three families - frozen head, LoRA, full fine-tune - so
no config gets a different code path. The backbone runs LIVE in every case,
including frozen ones that could have used the cache, because identical
evaluation code across configs matters more here than a few minutes saved.

Protocol: min_area is swept on VAL for each config independently and the
winner applied to TEST. Choosing it on test would be selecting on test.
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor, AutoConfig

from spatial_head import build, N_CLASSES
from train_lora import MODEL_ID, DATA_DIR, MASK_DIR, SPLIT_DIR, CKPT_DIR, GRID

CLASS_NAMES = ["background", "defect_1", "defect_2", "defect_3", "defect_4"]

CONFIGS = [
    ("B4_conv_subpatch_dice", "spatial_conv_up4_wp0.3_dice1.0_C.pt"),
    ("B5_frozen_wide_head",   "spatial_conv_up4_h1024_wp0.3_dice1.0_ep30.pt"),
    ("B6_lora_r16",           "lora_r16_h1024_wp0.3_full.pt"),
    ("B7_fullft_lr1e-5",      "fullft_lr1e-05_g0.85_h1024_b.pt"),
    ("B7_fullft_lr5e-5",      "fullft_lr5e-05_g0.85_h1024_a.pt"),
    ("B7_fullft_lr1e-4",      "fullft_lr0.0001_g0.85_h1024_c.pt"),
    ("B8_fullft_aug_5e-5",    "fullft_lr5e-05_g0.85_h1024_aug.pt"),
    ("B9_fullft_aug_1e-4",    "fullft_lr0.0001_g0.85_h1024_aug1e4.pt"),
]
MIN_AREAS = [0, 200, 400, 800, 1600, 3200]


def build_model(ck, device):
    """Returns (backbone, head, taps, up) for any checkpoint family."""
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    if "backbone" in ck:                                    # full fine-tune
        model.load_state_dict(ck["backbone"])
        taps = tuple(ck["tap_layers"])
        head = build("conv", dim=384 * len(taps), upscale=4, hidden=ck["hidden"])
        head.load_state_dict(ck["head"]); up = 4

    elif "lora" in ck:                                      # LoRA adapters
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=ck["rank"], lora_alpha=ck["alpha"], lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj"])
        model = get_peft_model(model, cfg).to(device).eval()
        model.load_state_dict(ck["lora"], strict=False)
        taps = tuple(ck["tap_layers"])
        head = build("conv", dim=384 * len(taps), upscale=4, hidden=ck["hidden"])
        head.load_state_dict(ck["head"]); up = 4

    else:                                                   # frozen head on stock DINOv3
        dim = ck["dim"]
        taps = (2, 5, 8, 11) if dim == 384 * 4 else (11,)
        head = build(ck["kind"], dim=dim, upscale=ck["upscale"], hidden=ck.get("hidden") or 256)
        head.load_state_dict(ck["state_dict"]); up = 16 // ck["upscale"]

    return model, head.to(device).eval(), taps, up


@torch.no_grad()
def collect(ckpt_name, split, device, bs=4):
    """One pass. Returns per-(image,class) predicted area, GT area, intersection."""
    ck = torch.load(f"{CKPT_DIR}/{ckpt_name}", map_location=device, weights_only=False)
    model, head, taps, up = build_model(ck, device)
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    mean = np.array(proc.image_mean, dtype=np.float32)
    std = np.array(proc.image_std, dtype=np.float32)
    # read this off the config directly: a plain HF model also exposes
    # .base_model (the encoder), so that is not a reliable PEFT test
    n_special = 1 + getattr(AutoConfig.from_pretrained(MODEL_ID), "num_register_tokens", 0)
    ids = pd.read_csv(f"{SPLIT_DIR}/{split}_ids.csv")["ImageId"].tolist()

    P = np.zeros((len(ids), N_CLASSES), np.int64)
    G = np.zeros((len(ids), N_CLASSES), np.int64)
    I = np.zeros((len(ids), N_CLASSES), np.int64)

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
        out = model(pixel_values=x, output_hidden_states=True)
        parts = []
        for i in taps:
            h = out.hidden_states[i + 1][:, n_special:, :]
            parts.append(F.layer_norm(h, (h.shape[-1],)))
        tok = torch.cat(parts, dim=-1)
        feat = tok.permute(0, 2, 1).reshape(len(sel), -1, GRID[0], GRID[1])
        cells = head(feat).float().argmax(1).cpu().numpy().astype(np.uint8)

        for k in range(len(sel)):
            pred = np.repeat(np.repeat(cells[k], up, 0), up, 1)
            gt = masks[k]
            for c in range(N_CLASSES):
                pc, gc = pred == c, gt == c
                P[i0 + k, c] = pc.sum(); G[i0 + k, c] = gc.sum()
                I[i0 + k, c] = np.logical_and(pc, gc).sum()
    del model, head
    torch.cuda.empty_cache()
    return P, G, I


def score(P, G, I, min_area=0):
    Pm, Im = P.copy(), I.copy()
    if min_area:
        drop = Pm[:, 1:] < min_area
        Pm[:, 1:][drop] = 0
        Im[:, 1:][drop] = 0
    dice = np.array([(2 * Im[:, c].sum() / (Pm[:, c].sum() + G[:, c].sum()))
                     if (Pm[:, c].sum() + G[:, c].sum()) > 0 else np.nan
                     for c in range(N_CLASSES)])
    pair = []
    for c in range(1, N_CLASSES):
        p, g, i = Pm[:, c], G[:, c], Im[:, c]
        both = (p == 0) & (g == 0)
        one = (p == 0) ^ (g == 0)
        ok = ~both & ~one
        pair += [np.ones(both.sum()), np.zeros(one.sum()),
                 2 * i[ok] / np.maximum(p[ok] + g[ok], 1)]
    return dice, np.concatenate(pair).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="subset of config names")
    args = ap.parse_args()
    device = "cuda"
    todo = [(n, c) for n, c in CONFIGS if not args.only or n in args.only]

    rows = []
    for name, ckpt in todo:
        print(f"\n### {name}  ({ckpt})", flush=True)
        Pv, Gv, Iv = collect(ckpt, "val", device)
        best_ma, best_k = 0, -1
        for ma in MIN_AREAS:                      # choose min_area on VAL
            _, k = score(Pv, Gv, Iv, ma)
            if k > best_k:
                best_ma, best_k = ma, k
        Pt, Gt, It = collect(ckpt, "test", device)
        # cache raw areas so any future re-scoring costs nothing
        np.savez_compressed(f"{CKPT_DIR}/../areas_{name}_test.npz", P=Pt, G=Gt, I=It)
        dice0, kag0 = score(Pt, Gt, It, 0)
        dice, kag = score(Pt, Gt, It, best_ma)
        print(f"  val-chosen min_area={best_ma} (val kaggle {best_k:.4f}) | "
              f"test mean {np.nanmean(dice0[1:]):.4f} (filtered {np.nanmean(dice[1:]):.4f}) | "
              f"test kaggle {kag0:.4f} -> {kag:.4f}", flush=True)
        rows.append((name, dice0, dice, kag, kag0, best_ma))

    print("\n\n" + "=" * 104)
    print("TEST SPLIT - per-class Dice (global accumulation) and the Kaggle metric")
    print("=" * 104)
    print("ALL per-class Dice below are UNFILTERED (min_area=0). The filter is reported")
    print("separately so it can never be silently mixed into a per-class number.\n")
    print(f"{'config':<24}{'c1':>8}{'c2':>8}{'c3':>8}{'c4':>8}{'mean1-4':>10}"
          f"{'mean*':>9}{'kaggle@0':>11}{'kaggle*':>10}{'min_area':>10}")
    print("-" * 104)
    for name, d0, df, kag, kag0, ma in sorted(rows, key=lambda r: np.nanmean(r[1][1:])):
        print(f"{name:<24}{d0[1]:>8.4f}{d0[2]:>8.4f}{d0[3]:>8.4f}{d0[4]:>8.4f}"
              f"{np.nanmean(d0[1:]):>10.4f}{np.nanmean(df[1:]):>9.4f}"
              f"{kag0:>11.4f}{kag:>10.4f}{ma:>10}")
    print("-" * 104)
    print("mean* / kaggle* = at the min_area chosen on VAL for that config (never on test)")


if __name__ == "__main__":
    main()
