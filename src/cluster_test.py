"""Does 5th place's near-duplicate premise hold on OUR data?

5th place noticed Severstal contains near-duplicate images from the same
manufacturing run and argued their labels are correlated, so the empty/non-empty
decision should be made per CLUSTER rather than per image. He clustered the test
features with k-means and averaged max-probabilities within each cluster before
thresholding.

The whole method rests on one assumption: cluster members share labels. That is
directly checkable on labelled data, and it must be checked first - if clusters
are label-MIXED, averaging would drag true positives down toward their clean
neighbours and actively hurt.

Measured here on val + holdout:
  - label homogeneity within clusters, against a shuffled baseline. Beating the
    baseline is the whole question; matching it means clustering found nothing
    labels care about.
  - how much a cluster-averaged decision would change vs per-image decisions
  - whether those changes are net right or wrong, scored on the real metric
"""
import argparse
import numpy as np
import pandas as pd
import cv2
import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoImageProcessor
from sklearn.cluster import KMeans

from spatial_head import N_CLASSES
from train_lora import MODEL_ID, DATA_DIR, MASK_DIR, SPLIT_DIR, CKPT_DIR

ENS = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\ensemble"
CLASSES = (1, 2, 3, 4)


@torch.no_grad()
def image_features(ckpt_name, split, device, bs=8):
    """Pooled backbone features per image - what the clustering runs on.
    5th place used the encoder's bottleneck output for the same reason: low
    dimension, and it captures overall appearance rather than defect detail."""
    path = f"{ENS}/feat_{split}.npy"
    if os.path.exists(path):
        return np.load(path)
    ck = torch.load(f"{CKPT_DIR}/{ckpt_name}", map_location=device, weights_only=False)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    model.load_state_dict(ck["backbone"])
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    mean = np.array(proc.image_mean, np.float32); std = np.array(proc.image_std, np.float32)
    ns = 1 + getattr(AutoConfig.from_pretrained(MODEL_ID), "num_register_tokens", 0)
    ids = pd.read_csv(f"{SPLIT_DIR}/{split}_ids.csv")["ImageId"].tolist()
    out = np.zeros((len(ids), 384), np.float32)
    for i0 in range(0, len(ids), bs):
        ch = ids[i0:i0 + bs]
        x = np.stack([(((cv2.imread(f"{DATA_DIR}/train_images/{f}")[:, :, ::-1]
                         .astype(np.float32) / 255.0) - mean) / std).transpose(2, 0, 1) for f in ch])
        h = model(pixel_values=torch.from_numpy(x).to(device)).last_hidden_state
        out[i0:i0 + len(ch)] = h[:, ns:, :].mean(1).float().cpu().numpy()
        if (i0 + bs) % 800 < bs:
            print(f"    {min(i0+bs,len(ids))}/{len(ids)}", flush=True)
    np.save(path, out)
    del model; torch.cuda.empty_cache()
    return out


def labels_for(split):
    ids = pd.read_csv(f"{SPLIT_DIR}/{split}_ids.csv")["ImageId"].tolist()
    Y = np.zeros((len(ids), N_CLASSES), bool)
    for i, image_id in enumerate(ids):
        m = cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png", cv2.IMREAD_UNCHANGED)
        for c in CLASSES:
            Y[i, c] = (m == c).any()
    return Y, ids


def homogeneity(labels, Y):
    """Mean over clusters of the majority-label fraction, per class. 1.0 means
    every cluster is internally unanimous."""
    sc = []
    for c in CLASSES:
        vals = []
        for k in np.unique(labels):
            mem = Y[labels == k, c]
            if len(mem) > 1:
                vals.append(max(mem.mean(), 1 - mem.mean()))
        sc.append(np.mean(vals))
    return np.array(sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="fullft_lr0.0001_g0.85_h1024_12k_ep16.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int, nargs="+", default=[100, 250, 500, 900])
    a = ap.parse_args()
    device = "cuda"

    print(f"### features for {a.split}")
    X = image_features(a.ckpt, a.split, device)
    Y, ids = labels_for(a.split)
    n = len(X)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    print(f"{n} images, {X.shape[1]}-dim features")
    print(f"positive rate per class: {[f'{Y[:,c].mean()*100:.1f}%' for c in CLASSES]}\n")

    print(f"{'k':>6}{'avg size':>10}   {'label homogeneity within clusters':<40}{'shuffled baseline':<22}")
    for k in a.k:
        if k >= n:
            continue
        km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(Xn)
        h = homogeneity(km.labels_, Y)
        rng = np.random.default_rng(0)
        base = np.mean([homogeneity(rng.permutation(km.labels_), Y) for _ in range(3)], axis=0)
        gain = h - base
        print(f"{k:>6}{n/k:>10.1f}   c1 {h[0]:.3f} c2 {h[1]:.3f} c3 {h[2]:.3f} c4 {h[3]:.3f}   "
              f"base c3 {base[2]:.3f}   c3 gain {gain[2]:+.3f}")

    print("\nInterpretation: clustering is only useful if homogeneity BEATS the shuffled")
    print("baseline. c3 is the informative class (41% positive, so the baseline sits near")
    print("0.6); c1/c2/c4 are rare enough that any clustering looks homogeneous by default.")


if __name__ == "__main__":
    main()
