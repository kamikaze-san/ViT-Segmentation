"""Does a VLM encoder add presence information on top of DINOv3? Uses the holdout presence scores saved by
encoder_probe.py. Combined = mean of rank-normalised scores (no fitting, so no holdout overfit). Also the rank
correlation of the two encoders' scores on negatives and positives: low correlation = different mistakes."""
import json, os
import numpy as np, pandas as pd, cv2
from scipy.stats import spearmanr

from train_lora import MASK_DIR, SPLIT_DIR
from vlm_score import auc

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\encoder_probe"


def ranks(x):
    return np.argsort(np.argsort(x, axis=0), axis=0) / (len(x) - 1)


def main(other="qwenvit", base="dinov3"):
    ho = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    y = np.array([[(cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED) == c).any()
                   for c in range(1, 5)] for f in ho])
    d, o = np.load(os.path.join(OUT, f"{base}_holdout_presence.npy")), np.load(os.path.join(OUT, f"{other}_holdout_presence.npy"))
    comb = (ranks(d) + ranks(o)) / 2
    res = {}
    for c in range(4):
        yc = y[:, c]
        res[f"c{c + 1}"] = {"positives": int(yc.sum()), "auc_dinov3": round(auc(d[:, c], yc), 4),
                            f"auc_{other}": round(auc(o[:, c], yc), 4), "auc_combined": round(auc(comb[:, c], yc), 4),
                            "rank_corr_on_negatives": round(float(spearmanr(d[~yc, c], o[~yc, c])[0]), 3),
                            "rank_corr_on_positives": round(float(spearmanr(d[yc, c], o[yc, c])[0]), 3) if yc.sum() > 2 else None}
    ya = y.any(1)                      # any-defect: max over classes per encoder first, then rank-average
    da, oa = d.max(1), o.max(1)
    res["any"] = {"auc_dinov3": round(auc(da, ya), 4), f"auc_{other}": round(auc(oa, ya), 4),
                  "auc_combined": round(auc((ranks(da) + ranks(oa)) / 2, ya), 4),
                  "rank_corr_on_negatives": round(float(spearmanr(da[~ya], oa[~ya])[0]), 3)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, f"compare_{base}_{other}.json"), "w"), indent=1)


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:3] if len(sys.argv) > 2 else ["qwenvit"]))
