"""Evaluate the training-free prototype/cosine-similarity segmentation for
class 2 on a given split, using the same global-accumulation Dice as
evaluate.py for direct comparability with the trained-decoder results.
"""
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from transformers import AutoImageProcessor, AutoModel

from rle_utils import rle_decode
from zero_shot_probe import DATA_DIR, MODEL_ID, IMG_SHAPE
from prototype_segmentation import predict_mask, TARGET_CLASS

SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=None, help="Limit number of images (for quick checks)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} ...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()

    defect_proto = np.load("prototype_class2_defect.npy")
    bg_proto = np.load("prototype_class2_bg.npy")
    threshold = float(np.load("prototype_class2_threshold.npy")[0])
    print(f"Using calibrated threshold: {threshold:.5f}")

    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    ids = pd.read_csv(f"{SPLIT_DIR}/{args.split}_ids.csv")["ImageId"].tolist()
    if args.n:
        ids = ids[:args.n]

    rows_by_image = df[df.ClassId == TARGET_CLASS].set_index("ImageId")["EncodedPixels"].to_dict()

    intersection, pred_sum, gt_sum = 0, 0, 0
    for i, image_id in enumerate(ids):
        img_path = f"{DATA_DIR}/train_images/{image_id}"
        image_bgr = cv2.imread(img_path)

        gt = rle_decode(rows_by_image[image_id], shape=IMG_SHAPE) if image_id in rows_by_image else np.zeros(IMG_SHAPE, dtype=np.uint8)
        pred = predict_mask(model, processor, device, image_bgr, defect_proto, bg_proto, IMG_SHAPE, threshold=threshold)

        intersection += np.logical_and(pred == 1, gt == 1).sum()
        pred_sum += (pred == 1).sum()
        gt_sum += (gt == 1).sum()

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(ids)}")

    dice = (2 * intersection / (pred_sum + gt_sum)) if (pred_sum + gt_sum) > 0 else float("nan")
    print(f"\nPrototype segmentation for class {TARGET_CLASS} on {args.split} split ({len(ids)} images):")
    print(f"  Dice: {dice:.4f}")
    print(f"  pred_sum={pred_sum}, gt_sum={gt_sum}, intersection={intersection}")


if __name__ == "__main__":
    main()
