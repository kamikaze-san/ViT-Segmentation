"""Compute expected per-class exposure per epoch under a proposed
WeightedRandomSampler, before actually running training with it.

Weight per image = 1 / (count of its rarest present class), background-only
images get a neutral baseline weight of 1.0.
"""
import pandas as pd

DATA_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"


def main():
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    train_ids = pd.read_csv(f"{SPLIT_DIR}/train_ids.csv")["ImageId"].tolist()
    train_ids_set = set(train_ids)

    train_df = df[df.ImageId.isin(train_ids_set)]
    classes_per_image = train_df.groupby("ImageId")["ClassId"].apply(set).to_dict()

    class_counts = {c: train_df[train_df.ClassId == c].ImageId.nunique() for c in [1, 2, 3, 4]}
    n_bg = len(train_ids) - len(classes_per_image)
    class_counts["bg"] = n_bg  # treat background as just another class, same 1/count scale
    print("Original per-class image counts (train split):")
    for c in [1, 2, 3, 4]:
        print(f"  class {c}: {class_counts[c]}")
    print(f"  background-only: {n_bg}")
    print(f"  total train images: {len(train_ids)}\n")

    weights = {}
    for image_id in train_ids:
        classes = classes_per_image.get(image_id)
        group_counts = [class_counts[c] for c in classes] if classes else [class_counts["bg"]]
        weights[image_id] = 1.0 / min(group_counts)

    total_weight = sum(weights.values())
    probs = {k: w / total_weight for k, w in weights.items()}

    num_samples = len(train_ids)  # keep epoch length the same

    print(f"{'Class':>15} {'Original count':>15} {'Expected/epoch':>16} {'Multiplier':>11}")
    for c in [1, 2, 3, 4]:
        imgs_with_c = [iid for iid in train_ids if c in classes_per_image.get(iid, set())]
        expected = sum(probs[iid] for iid in imgs_with_c) * num_samples
        orig = class_counts[c]
        print(f"{'class ' + str(c):>15} {orig:>15} {expected:>16.0f} {expected/orig:>10.2f}x")

    bg_imgs = [iid for iid in train_ids if not classes_per_image.get(iid)]
    expected_bg = sum(probs[iid] for iid in bg_imgs) * num_samples
    print(f"{'background':>15} {n_bg:>15} {expected_bg:>16.0f} {expected_bg/n_bg:>10.2f}x")

    print(f"\nWeight range: min={min(weights.values()):.6f}, max={max(weights.values()):.6f}")
    print(f"Max/min weight ratio: {max(weights.values())/min(weights.values()):.1f}x")


if __name__ == "__main__":
    main()
