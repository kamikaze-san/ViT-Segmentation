"""Stratified train/val/test split by class-presence combination, grouped by
ImageId, over ALL images in train_images (including background-only images
that have no rows in train.csv at all).

Rare class-presence combos (each < ~50 images: (1,2), (2,3), (1,2,3), (2,4))
are merged into a single 'rare' stratification bucket so sklearn's stratified
split doesn't choke on combos with too few members to split three ways.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"

RARE_THRESHOLD = 50
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
RANDOM_STATE = 42


def build_strat_key(df, all_image_ids):
    grouped = df.groupby("ImageId")["ClassId"].apply(lambda x: tuple(sorted(x)))
    combo_counts = grouped.value_counts()
    rare_combos = set(combo_counts[combo_counts < RARE_THRESHOLD].index)

    keys = {}
    for img_id in all_image_ids:
        if img_id not in grouped.index:
            keys[img_id] = "bg"
        else:
            combo = grouped[img_id]
            keys[img_id] = "rare" if combo in rare_combos else "_".join(map(str, combo))
    return keys


def main():
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    all_image_ids = sorted(os.listdir(f"{DATA_DIR}/train_images"))

    keys = build_strat_key(df, all_image_ids)
    key_series = pd.Series(keys)
    print("Stratification bucket sizes:")
    print(key_series.value_counts())

    ids = key_series.index.to_numpy()
    strat = key_series.to_numpy()

    train_ids, temp_ids, train_strat, temp_strat = train_test_split(
        ids, strat, train_size=TRAIN_FRAC, stratify=strat, random_state=RANDOM_STATE
    )
    val_ids, test_ids = train_test_split(
        temp_ids,
        train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=temp_strat,
        random_state=RANDOM_STATE,
    )

    print(f"\ntrain: {len(train_ids)}  val: {len(val_ids)}  test: {len(test_ids)}")

    for name, split_ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        out_path = f"{OUT_DIR}/{name}_ids.csv"
        pd.DataFrame({"ImageId": split_ids}).to_csv(out_path, index=False)
        print(f"Saved {out_path}")

    # sanity check: verify per-bucket proportions held roughly in each split
    print("\nPer-bucket split check (train/val/test counts):")
    for bucket in key_series.unique():
        t = (key_series[train_ids] == bucket).sum()
        v = (key_series[val_ids] == bucket).sum()
        te = (key_series[test_ids] == bucket).sum()
        print(f"  {bucket:>10}: {t:>5} / {v:>4} / {te:>4}")


if __name__ == "__main__":
    main()
