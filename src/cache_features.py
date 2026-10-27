"""Cache DINOv3 patch features for the train split, at native resolution.

Keeps ALL defect patches (rare, precious) and a random subsample of
background patches per image (abundant), so the cache stays ~1-2 GB instead
of ~11 GB while losing no minority-class signal.

Run once; all downstream head experiments (linear, MLP, etc.) then train off
this cache in minutes instead of re-running DINOv3.
"""
import os
import numpy as np
import pandas as pd
import cv2

from dino_features import load_model, extract_patch_features, IMG_SHAPE, PATCH_SIZE

DATA_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
MASK_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\masks"
SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\feature_cache"

N_BG_PER_IMAGE = 150  # random background patches kept per image


def patch_labels_multiclass(mask, grid_shape):
    """Per-patch class label: the most frequent defect class present in the
    patch, or 0 if the patch contains no defect pixels at all."""
    gh, gw = grid_shape
    blocks = (mask.reshape(gh, PATCH_SIZE, gw, PATCH_SIZE)
                  .transpose(0, 2, 1, 3)
                  .reshape(gh * gw, PATCH_SIZE * PATCH_SIZE))
    counts = np.stack([(blocks == c).sum(axis=1) for c in [1, 2, 3, 4]], axis=1)
    has_defect = counts.sum(axis=1) > 0
    labels = np.zeros(gh * gw, dtype=np.uint8)
    labels[has_defect] = counts[has_defect].argmax(axis=1) + 1
    return labels


def main(split="train", layers=None, tag=""):
    os.makedirs(OUT_DIR, exist_ok=True)
    model, norm, device = load_model()
    print(f"Model on {device}; caching '{split}' split at native {IMG_SHAPE}; "
          f"layers={layers if layers else 'final only'}")

    ids = pd.read_csv(f"{SPLIT_DIR}/{split}_ids.csv")["ImageId"].tolist()
    rng = np.random.default_rng(0)

    feats_out, labels_out = [], []
    for i, image_id in enumerate(ids):
        image_bgr = cv2.imread(f"{DATA_DIR}/train_images/{image_id}")
        stem = os.path.splitext(image_id)[0]
        mask = cv2.imread(f"{MASK_DIR}/{stem}.png", cv2.IMREAD_UNCHANGED)

        tokens, grid = extract_patch_features(model, norm, device, image_bgr, layers=layers)
        labels = patch_labels_multiclass(mask, grid)

        keep = np.where(labels != 0)[0]
        bg = np.where(labels == 0)[0]
        if len(bg) > N_BG_PER_IMAGE:
            bg = rng.choice(bg, size=N_BG_PER_IMAGE, replace=False)
        keep = np.concatenate([keep, bg])

        feats_out.append(tokens[keep].astype(np.float16))
        labels_out.append(labels[keep])

        if (i + 1) % 500 == 0:
            n = sum(len(x) for x in labels_out)
            print(f"  {i + 1}/{len(ids)} images, {n} patches cached")

    feats = np.concatenate(feats_out)
    labels = np.concatenate(labels_out)
    np.save(f"{OUT_DIR}/{split}{tag}_feats.npy", feats)
    np.save(f"{OUT_DIR}/{split}{tag}_labels.npy", labels)

    print(f"\nSaved {feats.shape[0]} patches, dim {feats.shape[1]}, "
          f"{feats.nbytes / 1e9:.2f} GB")
    print("Per-class patch counts:")
    for c in range(5):
        print(f"  class {c}: {(labels == c).sum()}")


if __name__ == "__main__":
    import argparse
    from dino_features import DEFAULT_LAYERS
    ap = argparse.ArgumentParser()
    ap.add_argument("split", nargs="?", default="train")
    ap.add_argument("--multilayer", action="store_true",
                    help=f"concatenate blocks {DEFAULT_LAYERS} instead of final layer only")
    a = ap.parse_args()
    main(a.split,
         layers=DEFAULT_LAYERS if a.multilayer else None,
         tag="_ml" if a.multilayer else "")
