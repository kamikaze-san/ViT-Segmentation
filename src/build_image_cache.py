"""Pre-decode every image to a uint8 memmap so MIM training is GPU-bound.

JPEG decode in the training loop costs ~10 min per epoch out of ~23. Caching
removes it, taking the run from ~23 min/epoch to ~13.

Stored as a SINGLE channel: every Severstal image is exactly grayscale (checked
across 60 random images, max |R-G| = |G-B| = 0), so the three channels are
redundant. That takes the cache from 20.7 GB to 6.9 GB. Each image is verified
individually while building - if any image is genuinely colour the build says so
rather than silently discarding two channels.

Covers train_images AND the 5506 competition test_images. The test images are
the whole point of the exercise: masked pretraining is the only stage where the
model ever sees the target distribution, and it needs no labels to do it.
"""
import argparse
import os
import numpy as np
import cv2
import pandas as pd

DATA = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\image_cache"
H, W = 256, 1600


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    files = ([("train_images", f) for f in sorted(os.listdir(f"{DATA}/train_images"))] +
             [("test_images", f) for f in sorted(os.listdir(f"{DATA}/test_images"))])
    if a.limit:
        files = files[:a.limit]
    n = len(files)
    gb = n * H * W / 2**30
    print(f"{n} images -> ({n}, {H}, {W}) uint8 = {gb:.1f} GB")

    arr = np.lib.format.open_memmap(f"{OUT}/images.npy", mode="w+",
                                    dtype=np.uint8, shape=(n, H, W))
    rows, colour, wrong_size = [], 0, 0
    for i, (sub, f) in enumerate(files):
        img = cv2.imread(f"{DATA}/{sub}/{f}")
        if img.shape[:2] != (H, W):
            wrong_size += 1
            img = cv2.resize(img, (W, H))
        if not (np.array_equal(img[:, :, 0], img[:, :, 1])
                and np.array_equal(img[:, :, 1], img[:, :, 2])):
            colour += 1
        arr[i] = img[:, :, 0]          # channels verified identical; keep one
        rows.append({"ImageId": f, "source": sub, "index": i})
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n}", flush=True)
    arr.flush()
    pd.DataFrame(rows).to_csv(f"{OUT}/index.csv", index=False)

    print(f"\nwrote {OUT}/images.npy ({os.path.getsize(f'{OUT}/images.npy')/2**30:.1f} GB)")
    print(f"  train_images {sum(1 for s,_ in files if s=='train_images')} | "
          f"test_images {sum(1 for s,_ in files if s=='test_images')}")
    print(f"  non-grayscale images found: {colour}  (expected 0)")
    print(f"  wrong-size images resized : {wrong_size}  (expected 0)")
    if colour:
        print("  WARNING: some images were NOT grayscale - two channels were discarded")


if __name__ == "__main__":
    main()
