"""Two checks on the E26 soft ViT-B member, against vitb_lora_warm_pseudo01.

1. HOLDOUT (labels): does soft training draw better-sized masks on ViT-B, as it did on
   ViT-S (E25)? boundary_diag's per-pair analysis, each model at its own single-model
   rule (pseudo01 at the E17 PIX, soft01 at its rate-matched PIX). Neither model saw
   holdout. Holdout cannot rank models for Kaggle (E12) - mechanism only.

2. TEST (no labels): mask-level agreement. The presence-decision check could not see
   mask shape, which is what soft training changes. On the 2,104 test images outside
   the round-1 pseudo set, for every (image, class) pair BOTH models call positive:
   Dice between their full-resolution masks and their area ratio. Each candidate vs
   each of the other four V79 members, the candidates vs each other, and V79 vs the
   swapped ensemble. Kaggle-cell inference (3-view TTA, fp16, bilinear).
"""
import json, os, time
import numpy as np, pandas as pd, cv2, torch

import boundary_diag as BD
import pseudo_teacher as PT
from pseudo_diag import load
from train_lora import DATA_DIR, SPLIT_DIR

OUT = os.path.join(PT.OUTPUTS, "soft_member")
E17 = [0.60, 0.60, 0.60, 0.45]
MINA = PT.E17_MINA
MODELS = {"vitb":     ("kaggle_bundle_vitb/model_vitb.pt", E17),
          "vits_r16": ("kaggle_bundle_lora/model_vits_lora_r16.pt", E17),
          "lora_r8":  ("kaggle_bundle_lora/model_lora_r8.pt", E17),
          "bce16":    ("kaggle_bundle_bce/model_bce16.pt", [0.60, 0.40, 0.65, 0.55]),
          "pseudo01": ("kaggle_bundle_pseudo/model_vitb_lora_warm_pseudo01.pt", E17),
          "soft01":   ("kaggle_bundle_calib/model_vitb_lora_warm_soft_pseudo01.pt", [0.51, 0.43, 0.44, 0.32])}
OTHERS = ["vitb", "vits_r16", "lora_r8", "bce16"]
ENS = {"V79": (OTHERS + ["pseudo01"], [0.65, 0.60, 0.65, 0.58]),
       "V79_swap": (OTHERS + ["soft01"], [0.65, 0.60, 0.64, 0.60])}
ENS_MINA = [630, 1470, 1050, 3150]


def holdout_check():
    hold = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    res = {}
    print(f"=== 1. HOLDOUT ({len(hold)} images), each model at its own single-model rule, min_area {MINA}")
    print(f"{'model':<10}{'PIX':<26}{'kaggle':>8}{'FP':>7}{'FN':>7}{'mask':>7}   {'TP Dice c1-c4':<28}"
          f"{'area pred/label c1-c4':<26}{'overshoot share c1-c4'}")
    for name in ("pseudo01", "soft01"):
        rel, pix = MODELS[name]
        BD.PIX = pix
        net = PT.load_member(rel, "cuda")
        R = BD.run_set(net, hold, "cuda")
        P = R["n"] * 4
        dsum = sum(np.sum(R["d0"][j]) for j in range(4))
        o = {"pix": pix, "kaggle": round((R["ee"] + dsum) / P, 5),
             "loss_fp": round(R["fp"].sum() / P, 5), "loss_fn": round(R["fn"].sum() / P, 5),
             "loss_mask": round((R["tp"].sum() - dsum) / P, 5),
             "tp_dice": [round(float(np.mean(R["d0"][j])), 4) for j in range(4)],
             "area_ratio_median": [round(float(np.median(R["ratio"][j])), 3) for j in range(4)],
             "overshoot_share": [round(float(R["fpb"][j].sum() / max(R["fpb"][j].sum() + R["fnb"][j].sum(), 1)), 3) for j in range(4)]}
        res[name] = o
        print(f"{name:<10}{str(pix):<26}{o['kaggle']:>8.5f}{o['loss_fp']:>7.4f}{o['loss_fn']:>7.4f}{o['loss_mask']:>7.4f}   "
              f"{' '.join(f'{v:.3f}' for v in o['tp_dice']):<28}{' '.join(f'{v:.2f}' for v in o['area_ratio_median']):<26}"
              f"{' '.join(f'{100 * v:.0f}%' for v in o['overshoot_share'])}", flush=True)
        del net; torch.cuda.empty_cache()
    return res


def test_check():
    files = sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))
    pseudo = set(pd.read_csv(os.path.join(SPLIT_DIR, "pseudo_ids.csv"))["ImageId"])
    files = [f for f in files if f not in pseudo]
    nets = {m: PT.load_member(rel, "cuda") for m, (rel, _) in MODELS.items()}
    pairs = [(c, m) for c in ("pseudo01", "soft01") for m in OTHERS] + [("soft01", "pseudo01"), ("V79_swap", "V79")]
    acc = {pr: {"dice": [[] for _ in range(4)], "ratio": [[] for _ in range(4)]} for pr in pairs}
    ens_flip = np.zeros(4, int)
    t0 = time.time()
    for i0 in range(0, len(files), 4):
        x = load(os.path.join(DATA_DIR, "test_images"), files[i0:i0 + 4]).cuda()
        probs = {m: PT.member_probs(net, x, PT.TTA, half=True) for m, net in nets.items()}
        for e, (mem, _) in ENS.items():
            probs[e] = sum(probs[m] for m in mem) / len(mem)
        for k in range(len(x)):
            for j in range(4):
                masks = {}
                for m in list(MODELS) + list(ENS):
                    pix, mina = (MODELS[m][1], MINA) if m in MODELS else (ENS[m][1], ENS_MINA)
                    sel = cv2.resize(probs[m][k, j + 1], (1600, 256), interpolation=cv2.INTER_LINEAR) > pix[j]
                    masks[m] = sel if sel.sum() >= mina[j] else None
                ens_flip[j] += (masks["V79"] is None) != (masks["V79_swap"] is None)
                for a, b in pairs:
                    if masks[a] is not None and masks[b] is not None:
                        A, B = masks[a], masks[b]
                        acc[(a, b)]["dice"][j].append(2 * (A & B).sum() / (A.sum() + B.sum()))
                        acc[(a, b)]["ratio"][j].append(A.sum() / B.sum())
        if (i0 // 4) % 100 == 0:
            done = i0 + 4; el = time.time() - t0
            print(f"  test {min(done, len(files))}/{len(files)} | ETA {(len(files) - done) * el / done / 60:5.1f} min", flush=True)

    print(f"\n=== 2. TEST, {len(files)} images outside the round-1 pseudo set: mask Dice on pairs BOTH call positive")
    print(f"{'pair':<24}{'pairs c1-c4':<22}{'mask Dice c1-c4':<30}{'all':>6}{'area ratio (median) c1-c4':>30}")
    res = {}
    for pr in pairs:
        d, r = acc[pr]["dice"], acc[pr]["ratio"]
        n = [len(v) for v in d]
        allm = float(np.mean([x for v in d for x in v])) if sum(n) else float("nan")
        res[f"{pr[0]}_vs_{pr[1]}"] = {"n": n, "mask_dice": [round(float(np.mean(v)), 4) if v else None for v in d],
                                     "mask_dice_all": round(allm, 4),
                                     "area_ratio_median": [round(float(np.median(v)), 3) if v else None for v in r]}
        print(f"{pr[0] + ' vs ' + pr[1]:<24}{str(n):<22}{' '.join(f'{np.mean(v):.3f}' if v else '  -  ' for v in d):<30}"
              f"{allm:>6.3f}{' '.join(f'{np.median(v):.2f}' if v else ' - ' for v in r):>30}")
    for c in ("pseudo01", "soft01"):
        print(f"  mean mask Dice of {c} with the other 4 members: "
              f"{np.mean([res[f'{c}_vs_{m}']['mask_dice_all'] for m in OTHERS]):.4f}")
    res["V79_vs_swap_presence_flips"] = ens_flip.tolist()
    print(f"  V79 vs swap: presence decisions that differ per class {ens_flip.tolist()} of {len(files)} images")
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    res = {"holdout": holdout_check()}
    res["test"] = test_check()
    json.dump(res, open(os.path.join(OUT, "mask_agreement.json"), "w"), indent=1)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
