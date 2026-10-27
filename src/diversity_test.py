"""Does DISAGREEMENT actually explain the ensemble gains, or is that a story?

Every recent decision has rested on "diverse members help" - lora_r8 was kept
despite the holdout ranking it last, on the argument that being differently
trained (8.8k, unaugmented, adapters only) makes it disagree more. That argument
has never been checked against the disagreement it predicts.

Measures, for each candidate 5th member:
  - disagreement with the current 4-member ensemble on the empty/non-empty call,
    which is the decision the Kaggle metric actually scores
  - the holdout ensemble gain from adding it, already measured

If diversity drives the gains, those two should track. If they do not, the
principle is wrong and member choice needs a different basis.
"""
import numpy as np
import pandas as pd

from spatial_head import N_CLASSES
from train_lora import SPLIT_DIR

ENS = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\ensemble"
CLASSES = (1, 2, 3, 4)
PIX = [0.60, 0.60, 0.60, 0.45]
MINA = [1050, 2450, 1750, 5250]
BASE = ["vitb", "12k_ep16", "lora_r8", "bce16nt"]
# holdout gain from adding each as a 5th member, measured earlier
GAIN = {"lovasz": 0.00108, "vitl_lora": 0.00054, "strong16": 0.00054,
        "B9_mild": 0.00053, "vitl_bce": 0.00006, "vitb_lora": -0.00026,
        "vits_taps": -0.00037, "B7_noaug": -0.00121}


def decisions(names, ids):
    """Per (image,class) non-empty decision for the mean of `names`."""
    arrs = [np.load(f"{ENS}/probs_{n}_holdout.npy", mmap_mode="r") for n in names]
    out = np.zeros((len(ids), len(CLASSES)), bool)
    for i in range(len(ids)):
        p = sum(np.asarray(a[i], np.float32) for a in arrs) / len(arrs)
        for j, c in enumerate(CLASSES):
            sel = p[c] > PIX[j]
            out[i, j] = sel.sum() * 16 >= MINA[j]      # stride-4 cell -> 16 px
    return out


def main():
    ids = pd.read_csv(f"{SPLIT_DIR}/holdout_ids.csv")["ImageId"].tolist()
    base = decisions(BASE, ids)
    print(f"base 4-member fires on {100*base.mean():.2f}% of pairs\n")
    print(f"{'candidate':<12}{'disagree%':>11}{'holdout gain':>14}")
    print("-" * 37)
    rows = []
    for name, gain in sorted(GAIN.items(), key=lambda kv: -kv[1]):
        try:
            d = decisions([name], ids)
        except FileNotFoundError:
            print(f"{name:<12}{'(no cache)':>11}"); continue
        dis = (d != base).mean()
        rows.append((dis, gain))
        print(f"{name:<12}{100*dis:>10.2f}%{gain:>+14.5f}")
    if len(rows) > 2:
        x = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
        r = np.corrcoef(x, y)[0, 1]
        print(f"\ncorrelation(disagreement, gain) = {r:+.3f}  over {len(rows)} candidates")
        print("positive => diversity drives the gains, as assumed")
        print("~0 or negative => the diversity story does not hold and member")
        print("   choice has been resting on a rationalisation")


if __name__ == "__main__":
    main()
