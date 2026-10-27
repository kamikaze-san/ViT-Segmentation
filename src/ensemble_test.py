"""Does ensembling actually help, and do our models complement each other?

Ensembling is only worth its inference cost if the members make DIFFERENT
mistakes. Our checkpoints share a backbone family, a head architecture, a loss
and a dataset - the LR variants were already shown to be near-duplicates. What
differs meaningfully now is augmentation strength (none / mild / strong) and
training data (8.8k / 12k), so those are the axes worth testing.

Reports, on labelled val:
  - each model alone: per-class Dice, and the real Kaggle metric
  - every ensemble combination: same
  - pairwise DISAGREEMENT: what fraction of (image,class) empty/non-empty
    decisions the two models make differently. Low disagreement means averaging
    them cannot help much, whatever the score says.

Probabilities are cached to disk on the first pass so re-scoring combinations
costs nothing.
"""
import argparse
import itertools
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoImageProcessor

from spatial_head import build, N_CLASSES
from train_lora import MODEL_ID, DATA_DIR, MASK_DIR, SPLIT_DIR, CKPT_DIR, GRID

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\ensemble"

MODELS = {
    "strong16": "fullft_lr0.0001_g0.85_h1024_strongaug.pt",   # strong aug, 8.8k
    "12k_ep16": "fullft_lr0.0001_g0.85_h1024_12k_ep16.pt",    # strong aug, 12k
    "B9_mild":  "fullft_lr0.0001_g0.85_h1024_aug1e4.pt",      # mild aug,  8.8k
    "B7_noaug": "fullft_lr0.0001_g0.85_h1024_s2.pt",          # no aug,    8.8k
    "lovasz":   "fullft_lr2e-05_g0.85_h1024_lovasz.pt",       # Lovasz phase on 12k_ep16
    "blackout": "fullft_lr0.0001_g0.85_h1024_blackout_ep16.pt",# strong aug + defect blackout
    "vitb":     "fullft_lr0.0001_g0.85_h1024_vitb_blackout_ep16.pt",  # ViT-B/16, same recipe
}


@torch.no_grad()
def cache_probs(name, ckpt_name, split, device, bs=4):
    """Softmax probabilities at stride 4, fp16, cached to disk."""
    path = f"{OUT}/probs_{name}_{split}.npy"
    if os.path.exists(path):
        return path
    os.makedirs(OUT, exist_ok=True)
    ck = torch.load(f"{CKPT_DIR}/{ckpt_name}", map_location=device, weights_only=False)
    # Checkpoints written before the --model-id flag existed have no "model_id"
    # key and are all ViT-S; newer ones carry their own backbone, so a ViT-B
    # checkpoint loads here without the caller needing to know which it is.
    model_id = ck.get("model_id") or MODEL_ID
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    model.load_state_dict(ck["backbone"])
    taps = tuple(ck["tap_layers"])
    dim = model.config.hidden_size          # 384 for ViT-S, 768 for ViT-B
    head = build("conv", dim=dim * len(taps), upscale=4, hidden=ck["hidden"]).to(device).eval()
    head.load_state_dict(ck["head"])
    proc = AutoImageProcessor.from_pretrained(model_id)
    mean = np.array(proc.image_mean, np.float32); std = np.array(proc.image_std, np.float32)
    ns = 1 + getattr(AutoConfig.from_pretrained(model_id), "num_register_tokens", 0)
    print(f"    {ckpt_name} | {model_id} dim {dim} | taps {taps}", flush=True)

    ids = pd.read_csv(f"{SPLIT_DIR}/{split}_ids.csv")["ImageId"].tolist()
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16,
                                    shape=(len(ids), N_CLASSES, 64, 400))
    for i0 in range(0, len(ids), bs):
        ch = ids[i0:i0 + bs]
        x = np.stack([(((cv2.imread(f"{DATA_DIR}/train_images/{f}")[:, :, ::-1]
                         .astype(np.float32) / 255.0) - mean) / std).transpose(2, 0, 1) for f in ch])
        o = model(pixel_values=torch.from_numpy(x).to(device), output_hidden_states=True)
        tok = torch.cat([F.layer_norm(o.hidden_states[t + 1][:, ns:, :],
                                      (o.hidden_states[t + 1].shape[-1],)) for t in taps], -1)
        p = head(tok.permute(0, 2, 1).reshape(len(ch), -1, GRID[0], GRID[1])).float().softmax(1)
        arr[i0:i0 + len(ch)] = p.cpu().numpy().astype(np.float16)
        if (i0 + bs) % 800 < bs:
            print(f"    {min(i0+bs,len(ids))}/{len(ids)}", flush=True)
    arr.flush(); del model, head; torch.cuda.empty_cache()
    return path


def score(prob_paths, ids, weights=None):
    """Average probabilities across models, then argmax. Returns per-class Dice,
    the Kaggle metric, and the per-(image,class) non-empty decision matrix."""
    arrs = [np.load(p, mmap_mode="r") for p in prob_paths]
    w = np.ones(len(arrs)) / len(arrs) if weights is None else np.array(weights) / np.sum(weights)
    inter = np.zeros(N_CLASSES, np.int64); psum = np.zeros(N_CLASSES, np.int64)
    gsum = np.zeros(N_CLASSES, np.int64)
    pair, decision = [], np.zeros((len(ids), N_CLASSES), bool)
    for i, image_id in enumerate(ids):
        p = sum(wi * np.asarray(a[i], np.float32) for wi, a in zip(w, arrs))
        cells = p.argmax(0)
        pred = np.repeat(np.repeat(cells.astype(np.uint8), 4, 0), 4, 1)
        gt = cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png", cv2.IMREAD_UNCHANGED)
        for c in range(N_CLASSES):
            pc, gc = pred == c, gt == c
            inter[c] += np.logical_and(pc, gc).sum(); psum[c] += pc.sum(); gsum[c] += gc.sum()
            if c:
                decision[i, c] = pc.any()
                pa, ga = pc.sum(), gc.sum()
                pair.append(1.0 if (pa == 0 and ga == 0) else
                            (0.0 if (pa == 0 or ga == 0) else
                             2 * np.logical_and(pc, gc).sum() / (pa + ga)))
    dice = np.array([(2 * inter[c] / (psum[c] + gsum[c])) if (psum[c] + gsum[c]) > 0 else np.nan
                     for c in range(N_CLASSES)])
    return dice, float(np.mean(pair)), decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    device = "cuda"
    ids = pd.read_csv(f"{SPLIT_DIR}/{a.split}_ids.csv")["ImageId"].tolist()

    paths = {}
    for name, ck in MODELS.items():
        print(f"### caching {name}")
        paths[name] = cache_probs(name, ck, a.split, device)

    print(f"\n{'members':<40}{'c1':>8}{'c2':>8}{'c3':>8}{'c4':>8}{'meanDice':>10}{'KAGGLE':>9}")
    print("-" * 91)
    results, decisions = {}, {}
    for name in MODELS:
        d, k, dec = score([paths[name]], ids)
        results[name] = (np.nanmean(d[1:]), k); decisions[name] = dec
        print(f"{name:<40}{d[1]:>8.4f}{d[2]:>8.4f}{d[3]:>8.4f}{d[4]:>8.4f}"
              f"{np.nanmean(d[1:]):>10.4f}{k:>9.5f}")
    print("-" * 91)
    best_single = max(results.values(), key=lambda v: v[1])[1]

    names = list(MODELS)
    for r in (2, 3, 4):
        for combo in itertools.combinations(names, r):
            d, k, _ = score([paths[n] for n in combo], ids)
            flag = "  <- beats best single" if k > best_single else ""
            print(f"{'+'.join(combo):<40}{d[1]:>8.4f}{d[2]:>8.4f}{d[3]:>8.4f}{d[4]:>8.4f}"
                  f"{np.nanmean(d[1:]):>10.4f}{k:>9.5f}{flag}")

    print("\nPAIRWISE DISAGREEMENT on the empty/non-empty decision")
    print("(fraction of the 4x N image-class pairs where the two models differ;")
    print(" low disagreement => averaging them cannot help much)")
    for x, y in itertools.combinations(names, 2):
        dis = (decisions[x][:, 1:] != decisions[y][:, 1:]).mean()
        print(f"  {x:<12} vs {y:<12} {100*dis:5.2f}%")


if __name__ == "__main__":
    main()
