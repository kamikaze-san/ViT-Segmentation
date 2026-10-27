"""Training-free prototype/cosine-similarity segmentation for class 2, using
frozen DINOv3 patch features. Zero trainable parameters:
  1. Build a "class 2" prototype = mean feature vector over all class-2-labeled
     patches in the 173 train-split images containing class 2.
  2. Build a "not class 2" prototype = mean feature vector over all other
     patches in those same images.
  3. For any image, classify each patch by nearest prototype (cosine sim).
  4. Upsample the patch-grid decision (nearest-neighbor) to full resolution.

Directly tests whether the demonstrated frozen-feature separability (0.935
AUC) is spatially exploitable into real masks, without any training-pipeline
confounds (loss weighting, checkpoint selection, Hungarian matching, etc).
"""
import numpy as np
import pandas as pd
import torch
import cv2
from transformers import AutoImageProcessor, AutoModel

from rle_utils import rle_decode
from zero_shot_probe import get_patch_features, DATA_DIR, MODEL_ID, IMG_SHAPE

SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
TARGET_CLASS = 2


def patch_label_from_mask(mask, grid_shape):
    gh, gw = grid_shape
    mask_small = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return (mask_small > 0).astype(np.uint8).reshape(-1)


def build_prototypes(model, processor, device):
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    train_ids = set(pd.read_csv(f"{SPLIT_DIR}/train_ids.csv")["ImageId"])

    class_rows = df[(df.ClassId == TARGET_CLASS) & (df.ImageId.isin(train_ids))]
    print(f"Building prototypes from {len(class_rows)} train images containing class {TARGET_CLASS}")

    pos_feats, neg_feats = [], []
    for i, (_, row) in enumerate(class_rows.iterrows()):
        img_path = f"{DATA_DIR}/train_images/{row.ImageId}"
        image_bgr = cv2.imread(img_path)
        mask = rle_decode(row.EncodedPixels, shape=IMG_SHAPE)

        patch_tokens, grid_shape = get_patch_features(model, processor, image_bgr, device)
        labels = patch_label_from_mask(mask, grid_shape)

        pos_feats.append(patch_tokens[labels == 1])
        neg_feats.append(patch_tokens[labels == 0])

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(class_rows)}")

    pos_feats = np.concatenate(pos_feats, axis=0)
    neg_feats = np.concatenate(neg_feats, axis=0)
    print(f"Positive patches: {pos_feats.shape[0]}, negative patches: {neg_feats.shape[0]}")

    defect_proto = pos_feats.mean(axis=0)
    bg_proto = neg_feats.mean(axis=0)
    defect_proto /= np.linalg.norm(defect_proto)
    bg_proto /= np.linalg.norm(bg_proto)
    return defect_proto, bg_proto


def predict_mask(model, processor, device, image_bgr, defect_proto, bg_proto, out_shape, threshold=0.0):
    patch_tokens, grid_shape = get_patch_features(model, processor, image_bgr, device)
    norm_tokens = patch_tokens / np.linalg.norm(patch_tokens, axis=1, keepdims=True)

    sim_defect = norm_tokens @ defect_proto
    sim_bg = norm_tokens @ bg_proto
    pred_patch = ((sim_defect - sim_bg) > threshold).astype(np.uint8).reshape(grid_shape)

    pred_full = cv2.resize(pred_patch, (out_shape[1], out_shape[0]), interpolation=cv2.INTER_NEAREST)
    return pred_full


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    defect_proto, bg_proto = build_prototypes(model, processor, device)
    np.save("prototype_class2_defect.npy", defect_proto)
    np.save("prototype_class2_bg.npy", bg_proto)
    print("Saved prototype vectors.")


if __name__ == "__main__":
    main()
