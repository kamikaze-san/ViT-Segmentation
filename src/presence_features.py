"""Step 1+2: per-pair segmentation outcomes, and cached pooled features.

Writes two things per split:

  areas_<split>.npz      predicted / GT / intersection area per (image, class),
                         from which the exact false positives (G==0 & P>0) and
                         misses (G>0 & P==0) fall out. These are the pairs the
                         presence classifier has to fix.

  pooled_<split>.npz     [CLS || mean-pool || max-pool] = 4608 dims per image.
                         Max-pooling is not optional here: a small c1 defect
                         covers ~20 of 1600 patches, so mean-pool dilutes it 80x
                         and would systematically hide the rare classes we most
                         need the classifier to catch.

Both come from ONE forward pass, so the expensive part is paid once and every
later experiment on the head costs seconds.
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
from train_lora import MODEL_ID, DATA_DIR, MASK_DIR, SPLIT_DIR, CKPT_DIR, GRID

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\presence"


@torch.no_grad()
def run(ckpt_name, split, device, bs=4, stock=False):
    ck = torch.load(f"{CKPT_DIR}/{ckpt_name}", map_location=device, weights_only=False)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    if not stock:
        model.load_state_dict(ck["backbone"])
    taps = tuple(ck["tap_layers"])
    head = build("conv", dim=384 * len(taps), upscale=4, hidden=ck["hidden"]).to(device).eval()
    head.load_state_dict(ck["head"])

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    mean = np.array(proc.image_mean, np.float32)
    std = np.array(proc.image_std, np.float32)
    n_special = 1 + getattr(AutoConfig.from_pretrained(MODEL_ID), "num_register_tokens", 0)

    ids = pd.read_csv(f"{SPLIT_DIR}/{split}_ids.csv")["ImageId"].tolist()
    P = np.zeros((len(ids), N_CLASSES), np.int64)
    G = np.zeros((len(ids), N_CLASSES), np.int64)
    I = np.zeros((len(ids), N_CLASSES), np.int64)
    pooled = np.zeros((len(ids), 3 * 384 * len(taps)), np.float32)

    for i0 in range(0, len(ids), bs):
        chunk = ids[i0:i0 + bs]
        imgs, masks = [], []
        for image_id in chunk:
            img = cv2.imread(f"{DATA_DIR}/train_images/{image_id}")
            rgb = img[:, :, ::-1].astype(np.float32) / 255.0
            imgs.append(((rgb - mean) / std).transpose(2, 0, 1))
            masks.append(cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png",
                                    cv2.IMREAD_UNCHANGED))
        x = torch.from_numpy(np.stack(imgs)).to(device)
        out = model(pixel_values=x, output_hidden_states=True)
        parts, cls_parts = [], []
        for t in taps:
            h = out.hidden_states[t + 1]
            parts.append(F.layer_norm(h[:, n_special:, :], (h.shape[-1],)))
            cls_parts.append(F.layer_norm(h[:, 0, :], (h.shape[-1],)))
        tok = torch.cat(parts, dim=-1)                       # (B, 1600, 1536)
        feat = tok.permute(0, 2, 1).reshape(len(chunk), -1, GRID[0], GRID[1])

        pooled[i0:i0 + len(chunk)] = torch.cat(
            [torch.cat(cls_parts, dim=-1), tok.mean(1), tok.max(1).values],
            dim=-1).float().cpu().numpy()

        cells = head(feat).float().argmax(1).cpu().numpy().astype(np.uint8)
        for k in range(len(chunk)):
            pred = np.repeat(np.repeat(cells[k], 4, 0), 4, 1)
            gt = masks[k]
            for c in range(N_CLASSES):
                pc, gc = pred == c, gt == c
                P[i0 + k, c] = pc.sum(); G[i0 + k, c] = gc.sum()
                I[i0 + k, c] = np.logical_and(pc, gc).sum()
        if (i0 + bs) % 2000 < bs:
            print(f"    {min(i0+bs, len(ids))}/{len(ids)}", flush=True)
    del model, head
    torch.cuda.empty_cache()
    return P, G, I, pooled, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="fullft_lr0.0001_g0.85_h1024_strongaug.pt")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--stock", action="store_true", help="use UNfine-tuned DINOv3 for the pooled features")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    suffix = "_stock" if a.stock else ""

    for split in a.splits:
        print(f"### {split}")
        P, G, I, pooled, ids = run(a.ckpt, split, device, stock=a.stock)
        np.savez_compressed(f"{OUT}/areas_{split}{suffix}.npz", P=P, G=G, I=I,
                            ids=np.array(ids))
        np.savez_compressed(f"{OUT}/pooled_{split}{suffix}.npz", X=pooled, ids=np.array(ids))

        pf = P[:, 1:].ravel(); gf = G[:, 1:].ravel()
        fp = int(((gf == 0) & (pf > 0)).sum())
        miss = int(((gf > 0) & (pf == 0)).sum())
        hit = int(((gf > 0) & (pf > 0)).sum())
        print(f"  pairs {len(gf)} | empty-GT {(gf==0).sum()} | FALSE POS {fp} "
              f"({100*fp/max((gf==0).sum(),1):.1f}% of empty) | MISSES {miss} "
              f"({100*miss/max((gf>0).sum(),1):.1f}% of positive) | hits {hit}")
        print(f"  pooled {pooled.shape} -> {OUT}\n")


if __name__ == "__main__":
    main()
