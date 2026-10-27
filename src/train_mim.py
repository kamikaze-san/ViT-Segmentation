"""Stage 1: masked feature prediction (iBOT / data2vec style) on DINOv3.

Continued pretraining on 18,074 UNLABELLED images - 12,568 Severstal-train plus
the 5,506 competition test images. Those last ones are the reason this exists:
it is the only stage in the whole project where the model sees the target
distribution, and it needs no labels to do it.

Why masked prediction rather than DINO multi-crop:

  The published comparison on small corpora is one-sided - masked autoencoder
  methods beat contrastive ones as data shrinks (SparK vs MoCoV2, +9.4% AUC on
  the smallest task), and contrastive SSL is explicitly described as unsuitable
  for small datasets without bespoke tuning. The mechanism is supervision
  density: DINO extracts ~10 global comparisons per image, masked prediction
  extracts one target per masked patch - ~960 per image here. With 18k distinct
  images that ratio is the whole argument.

  It also drops everything that makes DINO fragile at small batch: no negatives,
  no batch statistics, no centering or sharpening schedules. And no multi-crop
  means our 6:1 aspect strips need no awkward anisotropic crop design.

Design choices:

  Block-wise masking, not random
      Random masking is far too easy on images - neighbouring patches leak the
      answer, so the model learns interpolation rather than structure.
      Contiguous blocks force it to actually model how steel texture continues.

  Predict FEATURES, not pixels
      Pixel reconstruction spends capacity on sensor noise and texture grain -
      precisely the nuisance variation we want the model to become invariant
      to. Teacher features are already abstracted past that.

  Initialised from STOCK DINOv3, not our fine-tuned model
      This is the standard three-phase pipeline (SSL -> domain-adapt -> supervised
      fine-tune). Running a masked objective on a segmentation-specialised model
      would fight the supervised adaptation rather than precede it.

  Low LR with layer-wise decay
      Adaptation, not training. The published cell-segmentation adaptation used
      2e-4, but on 130k images; at 18k a gentler touch is safer against washing
      out the pretrained features we are trying to build on.

Honest prior: the closest published test of this (DINOCell, 130k images) found
continued pretraining washed out almost entirely under downstream full
fine-tuning, showing real gains ONLY in the zero-shot / OOD setting. Our problem
IS the OOD setting, which is why this is worth running - but Stage 2's frozen
probe is a genuine gate, not a formality.
"""
import argparse
import copy
import math
import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

CACHE = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\image_cache"
DATA = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
CKPT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints"
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
GH, GW = 16, 100
N_PATCH = GH * GW
N_BLOCKS, MAX_DEPTH = 12, 13


def block_mask(rng, ratio, gh=GH, gw=GW):
    """BEiT-style block masking. Random patch masking is trivially solved by
    interpolating neighbours; contiguous blocks force real inpainting."""
    m = np.zeros((gh, gw), dtype=bool)
    target = int(ratio * gh * gw)
    guard = 0
    while m.sum() < target and guard < 200:
        guard += 1
        h = rng.integers(2, 7)
        w = rng.integers(4, 26)
        r = rng.integers(0, max(gh - h, 1))
        c = rng.integers(0, max(gw - w, 1))
        m[r:r + h, c:c + w] = True
    return m.reshape(-1)


def param_depth(name):
    if name.startswith("embeddings."):
        return 0
    if name.startswith("model.layer."):
        return int(name.split(".")[2]) + 1
    return MAX_DEPTH


def build_groups(model, base_lr, gamma, wd):
    g = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            g.setdefault((param_depth(n), p.ndim <= 1), []).append(p)
    return [{"params": ps, "lr": base_lr * (gamma ** (MAX_DEPTH - d)),
             "weight_decay": 0.0 if nd else wd} for (d, nd), ps in sorted(g.items())]


def augment(img, rng):
    """Light only - the masking IS the training signal here, and heavy
    augmentation would compete with it rather than add to it."""
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1]
    if rng.random() < 0.5:
        img = np.clip(img.astype(np.float32) * (1 + rng.uniform(-0.2, 0.2))
                      + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--bb-lr", type=float, default=5e-5)
    p.add_argument("--gamma", type=float, default=0.85)
    p.add_argument("--pred-lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--mask-ratio", type=float, default=0.6)
    p.add_argument("--ema", type=float, default=0.996)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    device = "cuda"

    # The uint8 cache makes this GPU-bound (~13 vs ~23 min/epoch) but costs 6.9 GB.
    # With the disk near full that is optional, so fall back to decoding JPEGs
    # each epoch - slower, but needs no extra space. Building the cache pushed
    # the disk to 98% and broke torch's DLL load (pagefile could not grow), so
    # the fallback is the default path here, not a nicety.
    if os.path.exists(f"{CACHE}/images.npy"):
        imgs = np.load(f"{CACHE}/images.npy", mmap_mode="r")
        srcs = pd.read_csv(f"{CACHE}/index.csv")["source"].tolist()
        files = None
        print(f"using uint8 cache {imgs.shape}")
    else:
        imgs = None
        tr = sorted(os.listdir(f"{DATA}/train_images"))
        te = sorted(os.listdir(f"{DATA}/test_images"))
        files = ([(f"{DATA}/train_images", f) for f in tr]
                 + [(f"{DATA}/test_images", f) for f in te])
        srcs = ["train_images"] * len(tr) + ["test_images"] * len(te)
        print(f"no uint8 cache - decoding JPEGs each epoch ({len(files)} images)")

    total = len(imgs) if imgs is not None else len(files)
    n = total if not args.max_images else min(args.max_images, total)
    print(f"{n} unlabelled images ({srcs.count('train_images')} train + "
          f"{srcs.count('test_images')} competition test)")

    def load_one(j):
        if imgs is not None:
            return np.asarray(imgs[j])
        d, f = files[j]
        return cv2.imread(f"{d}/{f}", cv2.IMREAD_GRAYSCALE)

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    mean = np.float32(proc.image_mean).mean()      # grayscale: channels identical
    std = np.float32(proc.image_std).mean()

    student = AutoModel.from_pretrained(MODEL_ID).to(device)
    teacher = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    for q in teacher.parameters():
        q.requires_grad_(False)
    predictor = nn.Sequential(nn.Linear(384, 384), nn.GELU(), nn.Linear(384, 384)).to(device)
    n_special = 1 + getattr(student.config, "num_register_tokens", 0)

    groups = build_groups(student, args.bb_lr, args.gamma, args.wd)
    groups.append({"params": list(predictor.parameters()), "lr": args.pred_lr,
                   "weight_decay": args.wd})
    opt = torch.optim.AdamW(groups)
    steps_per_epoch = math.ceil(n / args.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)
    rng = np.random.default_rng(0)
    print(f"mask ratio {args.mask_ratio} (block-wise) | bs {args.bs} | "
          f"{steps_per_epoch} steps/epoch | backbone {args.bb_lr:.1e} LLRD {args.gamma}")

    gstep = 0
    total_steps = args.epochs * steps_per_epoch
    for ep in range(args.epochs):
        student.train()
        order = rng.permutation(n)
        tot, nb, t0, cos_acc = 0.0, 0, time.time(), 0.0
        for i0 in range(0, n, args.bs):
            sel = np.sort(order[i0:i0 + args.bs])
            batch = np.stack([augment(load_one(j), rng) for j in sel])
            x = (batch.astype(np.float32) / 255.0 - mean) / std
            x = torch.from_numpy(x).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            m = torch.from_numpy(np.stack([block_mask(rng, args.mask_ratio)
                                           for _ in range(len(sel))])).to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = student(pixel_values=x, bool_masked_pos=m).last_hidden_state[:, n_special:, :]
                with torch.no_grad():
                    t = teacher(pixel_values=x).last_hidden_state[:, n_special:, :]
                    t = F.layer_norm(t, (t.shape[-1],))     # normalised targets: stabilises
                ps, ts = predictor(s)[m], t[m]
                loss = F.smooth_l1_loss(ps.float(), ts.float())
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(predictor.parameters()), args.clip)
            opt.step(); sched.step(); gstep += 1

            # EMA momentum ramps to 1.0 so the teacher stabilises as training proceeds
            mom = 1 - (1 - args.ema) * (math.cos(math.pi * gstep / total_steps) + 1) / 2
            with torch.no_grad():
                for pt, psd in zip(teacher.parameters(), student.parameters()):
                    pt.mul_(mom).add_(psd.detach(), alpha=1 - mom)
                cos_acc += F.cosine_similarity(ps.float(), ts.float(), dim=-1).mean().item()
            tot += loss.item(); nb += 1

        print(f"ep {ep+1:2d} | loss {tot/nb:.4f} | cos(pred,target) {cos_acc/nb:.4f} | "
              f"{time.time()-t0:5.0f}s | lr {sched.get_last_lr()[-1]:.2e}", flush=True)

        if (ep + 1) % args.save_every == 0 or ep + 1 == args.epochs:
            path = f"{CKPT_DIR}/mim_dinov3{args.tag}_ep{ep+1}.pt"
            torch.save({"backbone": {k: v.detach().cpu().clone()
                                     for k, v in student.state_dict().items()},
                        "predictor": predictor.state_dict(), "epoch": ep + 1,
                        "mask_ratio": args.mask_ratio, "bb_lr": args.bb_lr,
                        "loss": tot / nb, "cos": cos_acc / nb}, path)
            print(f"  saved {os.path.basename(path)}", flush=True)


if __name__ == "__main__":
    main()
