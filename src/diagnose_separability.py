"""Corrected per-class separability diagnostic.

Fixes two flaws in the earlier version:
  1. Features are extracted at NATIVE 256x1600 resolution (16x100 patch grid),
     not the processor's default 224x224 square squash (14x14 grid).
  2. Cross-validation splits by IMAGE, not by patch, so spatially-adjacent
     patches from one image can't leak between train and test folds.

Only train-split images are used; val/test are never touched.
"""
import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from rle_utils import rle_decode
from dino_features import load_model, extract_patch_features, patch_labels_from_mask, IMG_SHAPE

DATA_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
N_IMAGES_PER_CLASS = 40
N_FOLDS = 4


def main():
    model, norm, device = load_model()
    print(f"Model loaded on {device}. Extracting at native {IMG_SHAPE} resolution.\n")

    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    train_ids = set(pd.read_csv(f"{SPLIT_DIR}/train_ids.csv")["ImageId"])

    for cls in [1, 2, 3, 4]:
        rows = df[(df.ClassId == cls) & (df.ImageId.isin(train_ids))]
        rows = rows.sample(n=min(N_IMAGES_PER_CLASS, len(rows)), random_state=0)

        feats_per_image, labels_per_image = [], []
        for _, row in rows.iterrows():
            image_bgr = cv2.imread(f"{DATA_DIR}/train_images/{row.ImageId}")
            mask = rle_decode(row.EncodedPixels, shape=IMG_SHAPE)
            tokens, grid = extract_patch_features(model, norm, device, image_bgr)
            feats_per_image.append(tokens)
            labels_per_image.append(patch_labels_from_mask(mask, grid))

        n_img = len(feats_per_image)
        pos_frac = np.concatenate(labels_per_image).mean()

        # Image-level K-fold: whole images held out, no patch leakage.
        fold_aucs = []
        img_idx = np.arange(n_img)
        folds = np.array_split(img_idx, N_FOLDS)
        for f in range(N_FOLDS):
            test_imgs = set(folds[f].tolist())
            tr_i = [i for i in img_idx if i not in test_imgs]
            te_i = [i for i in img_idx if i in test_imgs]

            Xtr = np.concatenate([feats_per_image[i] for i in tr_i])
            ytr = np.concatenate([labels_per_image[i] for i in tr_i])
            Xte = np.concatenate([feats_per_image[i] for i in te_i])
            yte = np.concatenate([labels_per_image[i] for i in te_i])

            if ytr.sum() == 0 or yte.sum() == 0 or yte.sum() == len(yte):
                continue
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(Xtr, ytr)
            fold_aucs.append(roc_auc_score(yte, clf.decision_function(Xte)))

        print(f"class {cls}: {n_img} images, {len(np.concatenate(labels_per_image))} patches, "
              f"{100*pos_frac:.2f}% positive")
        print(f"          held-out-IMAGE AUC = {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f} "
              f"(folds: {', '.join(f'{a:.3f}' for a in fold_aucs)})\n")


if __name__ == "__main__":
    main()
