"""Slide-ready image assets (PNG, no text baked in; labels are set in the slides).

  testbed_gt.png        one strip per class with its ground-truth overlay (training images, for the dataset slide)
  hit_c{1,3,4}.png      640-px crops around a clean holdout hit: ground truth over prediction
  fail_{overshoot,wrongclass,miss}.png   same, for the three failure examples
  frozen_pca.png        the frozen-DINOv3 figure with its white margins trimmed
Same model, thresholds and inference as slide_overlays.py. Crops are centred on the union of GT and prediction.
"""
import os
import numpy as np, cv2, torch
from PIL import Image

from slide_overlays import REL, PCA_STRIPS, read, gt_mask, predict, overlay, load_member

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\figures\slides\assets"
FIG = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\figures\zero_shot_dinov3_probe.png"
HITS = {"hit_c1": "51d092947", "hit_c3": "4c3afd528", "hit_c4": "51097aa45"}
FAILS = {"fail_overshoot": "a835a0e50", "fail_wrongclass": "154e7e802", "fail_miss": "0429258ad"}
GAP = 10


def save(arr_rgb, name):
    Image.fromarray(arr_rgb).save(os.path.join(OUT, name))
    print("saved", name, arr_rgb.shape, flush=True)


def crop_pair(i, pm, width=640):
    img, gt = read(i), gt_mask(i)
    union = (gt > 0) | (pm > 0)
    cols = np.flatnonzero(union.any(0))
    c = int(cols.mean()) if len(cols) else 800
    if len(cols) and cols.max() - cols.min() > width:                        # spread-out defects: centre on the bulk
        c = int(np.median(cols))
    x0 = int(np.clip(c - width // 2, 0, 1600 - width))
    top, bot = overlay(img, gt)[:, x0:x0 + width], overlay(img, pm)[:, x0:x0 + width]
    sep = np.full((GAP, width, 3), 246, np.uint8)
    return np.vstack([top, sep, bot])


def main():
    os.makedirs(OUT, exist_ok=True)
    net = load_member(REL, "cuda")
    ids = list(HITS.values()) + list(FAILS.values())
    pm = predict(net, ids)
    for name, i in {**HITS, **FAILS}.items():
        save(crop_pair(i, pm[i]), name + ".png")
    rows = []
    for i in PCA_STRIPS:
        rows += [overlay(read(i), gt_mask(i)), np.full((GAP, 1600, 3), 246, np.uint8)]
    save(np.vstack(rows[:-1]), "testbed_gt.png")
    im = np.array(Image.open(FIG).convert("RGB"))
    nonwhite = np.flatnonzero((im < 250).any(2).any(1)), np.flatnonzero((im < 250).any(2).any(0))
    y0, y1, x0, x1 = nonwhite[0].min(), nonwhite[0].max() + 1, nonwhite[1].min(), nonwhite[1].max() + 1
    save(im[max(0, y0 - 20):y1 + 20, max(0, x0 - 20):x1 + 20], "frozen_pca.png")


if __name__ == "__main__":
    main()
