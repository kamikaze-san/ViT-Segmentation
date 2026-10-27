"""Did unbiased training actually remove the over-extension? (E25 mechanism check)

Runs boundary_diag's per-pair analysis on the 500 holdout images for the E23 control
and both calibration arms, each at its OWN rate-matched thresholds (control at its
standard PIX), and prints the quantities E24 diagnosed: predicted/label area, the
overshoot share of wrong pixels, true-positive Dice, and the loss split between
false-alarm pairs, missed pairs and mask quality.

Holdout cannot rank models for Kaggle (E12); this checks the MECHANISM only.
"""
import json, os
import numpy as np, pandas as pd

import boundary_diag as BD
import pseudo_teacher as PT
from train_lora import SPLIT_DIR

ARMS = [("control", "kaggle_bundle_simplex/model_vits_ctl10.pt", None),
        ("A soft", "kaggle_bundle_calib/model_vits_soft10.pt", "soft"),
        ("B metric", "kaggle_bundle_calib/model_vits_metric10.pt", "metric")]


def main():
    rm = json.load(open(os.path.join(PT.OUTPUTS, "calib_ab", "rate_match.json")))
    hold = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    out = {}
    print(f"holdout {len(hold)} images | min_area {BD.MINA}")
    print(f"{'arm':<10}{'PIX':<26}{'kaggle':>8}{'FP':>7}{'FN':>7}{'mask':>7}   "
          f"{'TP Dice c1-c4':<28}{'area pred/label c1-c4':<26}{'overshoot share c1-c4'}")
    for name, rel, key in ARMS:
        BD.PIX = rm["control_pix"] if key is None else rm[key]["matched_pix"]
        net = PT.load_member(rel, "cuda")
        R = BD.run_set(net, hold, "cuda")
        P = R["n"] * 4
        dsum = sum(np.sum(R["d0"][j]) for j in range(4))
        score = (R["ee"] + dsum) / P
        tpd = [float(np.mean(R["d0"][j])) if R["d0"][j] else float("nan") for j in range(4)]
        ratio = [float(np.median(R["ratio"][j])) if R["ratio"][j] else float("nan") for j in range(4)]
        over = [float(R["fpb"][j].sum() / max(R["fpb"][j].sum() + R["fnb"][j].sum(), 1)) for j in range(4)]
        out[name] = {"pix": BD.PIX, "kaggle": round(score, 5), "fp_pairs": int(R["fp"].sum()), "fn_pairs": int(R["fn"].sum()),
                     "loss_fp": round(R["fp"].sum() / P, 5), "loss_fn": round(R["fn"].sum() / P, 5),
                     "loss_mask": round((R["tp"].sum() - dsum) / P, 5), "tp_dice": [round(v, 4) for v in tpd],
                     "area_ratio_median": [round(v, 3) for v in ratio], "overshoot_share": [round(v, 3) for v in over]}
        o = out[name]
        print(f"{name:<10}{str([round(v, 2) for v in BD.PIX]):<26}{score:>8.5f}{o['loss_fp']:>7.4f}{o['loss_fn']:>7.4f}{o['loss_mask']:>7.4f}   "
              f"{' '.join(f'{v:.3f}' for v in tpd):<28}{' '.join(f'{v:.2f}' for v in ratio):<26}"
              f"{' '.join(f'{100 * v:.0f}%' for v in over)}", flush=True)
        del net
    json.dump(out, open(os.path.join(PT.OUTPUTS, "calib_ab", "mech_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
