"""Per-class breakdown of the earlier zero-shot signal check: is frozen
DINOv3's raw feature separability specifically worse for class 2 than for
classes 1, 3, 4 - independent of how many training images each class has?
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
N_SAMPLES_PER_CLASS = 15


def patch_label_from_mask(mask, grid_shape):
    gh, gw = grid_shape
    mask_small = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return (mask_small > 0).astype(np.uint8).reshape(-1)


def main():
    df = pd.read_csv(f"{DATA_DIR}/train.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    for cls in [1, 2, 3, 4]:
        rows = df[df.ClassId == cls].sample(n=min(N_SAMPLES_PER_CLASS, (df.ClassId == cls).sum()), random_state=0)

        all_feats, all_labels = [], []
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

        pca = PCA(n_components=N_COMPONENTS)
        proj = pca.fit_transform(feats)

        aucs = []
        for i in range(N_COMPONENTS):
            auc = roc_auc_score(labels, proj[:, i])
            aucs.append(max(auc, 1 - auc))

        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        cv_auc = cross_val_score(clf, proj, labels, cv=3, scoring="roc_auc")

        print(f"class {cls}: n_patches={feats.shape[0]}, defect_frac={100*labels.mean():.2f}%, "
              f"best_component_auc={max(aucs):.3f}, linear_probe_auc={cv_auc.mean():.3f}+/-{cv_auc.std():.3f}")


if __name__ == "__main__":
    main()
