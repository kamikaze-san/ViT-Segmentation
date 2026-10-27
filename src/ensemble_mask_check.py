"""Is V79 itself still over-extended at its tuned thresholds? (decides E26's next step)

The ensemble-tuned PIX (.65 vs the single-model .60) may already shrink the averaged
masks to label size, in which case converting more members to soft targets can only
shrink them further. Holdout (500 images, labels), Kaggle-cell inference (3-view TTA,
fp16, bilinear), V79 and the soft swap at their own tuned rules: predicted/label
area, overshoot share of wrong pixels, true-positive Dice, and the loss split.
Holdout cannot rank models for Kaggle (E12); this measures mask size only.
"""
import json, os
import numpy as np, pandas as pd, cv2

import pseudo_teacher as PT
from pseudo_diag import load
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR

MODELS = {"vitb":     "kaggle_bundle_vitb/model_vitb.pt",
          "vits_r16": "kaggle_bundle_lora/model_vits_lora_r16.pt",
          "lora_r8":  "kaggle_bundle_lora/model_lora_r8.pt",
          "bce16":    "kaggle_bundle_bce/model_bce16.pt",
          "pseudo01": "kaggle_bundle_pseudo/model_vitb_lora_warm_pseudo01.pt",
          "soft01":   "kaggle_bundle_calib/model_vitb_lora_warm_soft_pseudo01.pt"}
BASE = ["vitb", "vits_r16", "lora_r8", "bce16"]
ENS = {"V79":      (BASE + ["pseudo01"], [0.65, 0.60, 0.65, 0.58]),
       "V79_swap": (BASE + ["soft01"],   [0.65, 0.60, 0.64, 0.60])}
MINA = [630, 1470, 1050, 3150]


def main():
    ids = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    nets = {m: PT.load_member(rel, "cuda") for m, rel in MODELS.items()}
    st = {e: {"ee": 0, "fp": np.zeros(4, int), "fn": np.zeros(4, int), "d": [[] for _ in range(4)],
              "ratio": [[] for _ in range(4)], "over": np.zeros(4), "under": np.zeros(4)} for e in ENS}
    for i0 in range(0, len(ids), 4):
        ch = ids[i0:i0 + 4]
        x = load(os.path.join(DATA_DIR, "train_images"), ch).cuda()
        probs = {m: PT.member_probs(net, x, PT.TTA, half=True) for m, net in nets.items()}
        for k, f in enumerate(ch):
            g_all = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED)
            for e, (mem, pix) in ENS.items():
                avg = sum(probs[m][k] for m in mem) / len(mem)
                s = st[e]
                for j in range(4):
                    sel = cv2.resize(avg[j + 1], (1600, 256), interpolation=cv2.INTER_LINEAR) > pix[j]
                    g = g_all == j + 1
                    present, has = sel.sum() >= MINA[j], bool(g.any())
                    if not present and not has:
                        s["ee"] += 1
                    elif present and not has:
                        s["fp"][j] += 1
                    elif has and not present:
                        s["fn"][j] += 1
                    else:
                        s["d"][j].append(2 * (sel & g).sum() / (sel.sum() + g.sum()))
                        s["ratio"][j].append(sel.sum() / g.sum())
                        s["over"][j] += (sel & ~g).sum(); s["under"][j] += (g & ~sel).sum()
    P = len(ids) * 4
    res = {}
    print(f"holdout {len(ids)} images | min_area {MINA}")
    print(f"{'ensemble':<10}{'PIX':<26}{'kaggle':>8}{'FP':>7}{'FN':>7}{'mask':>7}   {'TP Dice c1-c4':<28}"
          f"{'area pred/label c1-c4':<26}{'overshoot share c1-c4'}")
    for e, (_, pix) in ENS.items():
        s = st[e]
        dsum = sum(np.sum(v) for v in s["d"]); tp = sum(len(v) for v in s["d"])
        o = {"pix": pix, "kaggle": round((s["ee"] + dsum) / P, 5), "loss_fp": round(s["fp"].sum() / P, 5),
             "loss_fn": round(s["fn"].sum() / P, 5), "loss_mask": round((tp - dsum) / P, 5),
             "tp_dice": [round(float(np.mean(v)), 4) for v in s["d"]],
             "area_ratio_median": [round(float(np.median(v)), 3) for v in s["ratio"]],
             "area_ratio_pooled": [round(float(np.sum(v)), 3) for v in [[0]]],
             "overshoot_share": [round(float(s["over"][j] / max(s["over"][j] + s["under"][j], 1)), 3) for j in range(4)]}
        o.pop("area_ratio_pooled")
        res[e] = o
        print(f"{e:<10}{str(pix):<26}{o['kaggle']:>8.5f}{o['loss_fp']:>7.4f}{o['loss_fn']:>7.4f}{o['loss_mask']:>7.4f}   "
              f"{' '.join(f'{v:.3f}' for v in o['tp_dice']):<28}{' '.join(f'{v:.2f}' for v in o['area_ratio_median']):<26}"
              f"{' '.join(f'{100 * v:.0f}%' for v in o['overshoot_share'])}", flush=True)
    json.dump(res, open(os.path.join(PT.OUTPUTS, "soft_member", "ensemble_mask_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
