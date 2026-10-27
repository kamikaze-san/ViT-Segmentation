"""Cache DINOv3 multi-layer features as FULL SPATIAL MAPS (grid layout kept).

The earlier cache_features.py flattened and subsampled patches, which is fine
for a per-patch head but destroys the neighbourhood structure a conv head or
a sub-patch decoder needs.

Written incrementally via open_memmap so the 43 GB train array is never held
in RAM (machine has 15.7 GB). Downstream training memory-maps it and reads
one image at a time (~4.9 MB, contiguous).

Layout: (n_images, 1600, 1536) fp16, patch order row-major over the 16x100 grid.
"""
import os
import numpy as np
import pandas as pd
import cv2
from numpy.lib.format import open_memmap

from dino_features import load_model, extract_patch_features, DEFAULT_LAYERS, IMG_SHAPE

DATA_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\spatial_cache"

N_PATCHES = 1600  # 16 x 100
DIM = 384 * len(DEFAULT_LAYERS)


def main(split):
    os.makedirs(OUT_DIR, exist_ok=True)
    ids = pd.read_csv(f"{SPLIT_DIR}/{split}_ids.csv")["ImageId"].tolist()

    feat_path = f"{OUT_DIR}/{split}_spatial.npy"
    gb = len(ids) * N_PATCHES * DIM * 2 / 1e9
    print(f"Caching '{split}': {len(ids)} images -> {feat_path}  ({gb:.1f} GB)")

    arr = open_memmap(feat_path, mode="w+", dtype=np.float16,
                      shape=(len(ids), N_PATCHES, DIM))

    model, norm, device = load_model()
    for i, image_id in enumerate(ids):
        img = cv2.imread(f"{DATA_DIR}/train_images/{image_id}")
        tokens, grid = extract_patch_features(model, norm, device, img, layers=DEFAULT_LAYERS)
        assert tokens.shape == (N_PATCHES, DIM), tokens.shape
        arr[i] = tokens.astype(np.float16)

        if (i + 1) % 500 == 0:
            arr.flush()
            print(f"  {i + 1}/{len(ids)}")

    arr.flush()
    del arr
    pd.DataFrame({"ImageId": ids}).to_csv(f"{OUT_DIR}/{split}_order.csv", index=False)
    print(f"Done: {split}, grid {grid}, dim {DIM}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
