"""Quantify whether ANY frozen DINOv3 PCA component correlates with defect
presence, beyond just the top-3 (used for RGB viz).

For each sample image: extract patch features, PCA to top-K components,
downsample the GT mask to patch-grid resolution (patch = "defect" if any
GT pixel inside it is defect), then compute per-component AUC separating
defect vs background patches. A component with AUC far from 0.5 means that
component's activation correlates with defect location, even if it wasn't
in the top-3 used for the earlier RGB visualization.

Still zero-shot / no model training - AUC is just a measurement, not a fit.
"""
import numpy as np
import pandas as pd
import torch
import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from transformers import AutoImageProcessor, AutoModel

from rle_utils import rle_decode
from zero_shot_probe import get_patch_features, DATA_DIR, MODEL_ID, IMG_SHAPE

N_COMPONENTS = 20
N_SAMPLES_PER_CLASS = 5


def patch_label_from_mask(mask, grid_shape):
    gh, gw = grid_shape
    mask_small = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return (mask_small > 0).astype(np.uint8).reshape(-1)  # any defect pixel in patch -> 1


def main():
    df = pd.read_csv(f"{DATA_DIR}/train.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    all_feats = []
    all_labels = []

    for cls in [1, 2, 3, 4]:
        rows = df[df.ClassId == cls].iloc[:N_SAMPLES_PER_CLASS]
        for _, row in rows.iterrows():
            img_path = f"{DATA_DIR}/train_images/{row.ImageId}"
            image_bgr = cv2.imread(img_path)
            mask = rle_decode(row.EncodedPixels, shape=IMG_SHAPE)

            patch_tokens, grid_shape = get_patch_features(model, processor, image_bgr, device)
            labels = patch_label_from_mask(mask, grid_shape)

            all_feats.append(patch_tokens)
            all_labels.append(labels)

    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"Total patches: {feats.shape[0]}, defect patches: {labels.sum()} ({100*labels.mean():.2f}%)")

    pca = PCA(n_components=N_COMPONENTS)
    proj = pca.fit_transform(feats)

    print(f"\nPer-component AUC (defect-patch vs background-patch), top {N_COMPONENTS} PCA components:")
    print(f"{'component':>10} {'explained_var%':>15} {'AUC':>8}")
    aucs = []
    for i in range(N_COMPONENTS):
        auc = roc_auc_score(labels, proj[:, i])
        auc_abs = max(auc, 1 - auc)  # component could correlate negatively
        aucs.append(auc_abs)
        print(f"{i:>10} {100*pca.explained_variance_ratio_[i]:>14.2f}% {auc_abs:>8.3f}")

    best = int(np.argmax(aucs))
    print(f"\nBest single component: #{best} with AUC={aucs[best]:.3f} (0.5 = no signal, 1.0 = perfect separation)")

    # also check a simple combined signal: logistic regression on all K components
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv_auc = cross_val_score(clf, proj, labels, cv=3, scoring="roc_auc")
    print(f"Linear probe (all {N_COMPONENTS} components, 3-fold CV) AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")


if __name__ == "__main__":
    main()
