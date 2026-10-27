"""Where does a single model lose the Kaggle metric, and would 'conservative edges' help?

Model: vitb_lora_warm (best clean single model, Kaggle 0.90424; never saw holdout or
test), Kaggle-cell inference (3-view TTA, fp16, bilinear), single-model rule.
Sets: the 500 holdout images, and 500 train12k images it WAS trained on.

Per set:
  A. Kaggle metric, decomposed: wrong presence (FP pair / FN pair, each costs 1)
     vs the Dice deficit on true-positive pairs.
  B. True-positive masks: predicted/GT area ratio, and where the wrong pixels sit
     by distance to the GT edge - overshoot (FP pixels outside GT) vs undershoot
     (FN pixels inside GT).
  C. Boundary-tolerant Dice: forgive wrong pixels within 1/2/4/8 px of the edge.
     How much of the deficit is 'a few pixels'.
  D. Post-hoc conservative edges: erode/dilate predicted masks by 1-4 px (presence
     unchanged). Retraining on eroded GT bakes in a bias this applies after the fact.
  E. Decoupled mask threshold: presence decided by the current rule, mask drawn at
     another pixel threshold. Dice theory: a pixel with P(defect) = q raises expected
     Dice iff q > Dice/2, so masks may want a LOWER threshold than presence does.
"""
import json, os, time
import numpy as np, pandas as pd, cv2

import pseudo_teacher as PT
from pseudo_diag import load
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR

OUT = os.path.join(PT.OUTPUTS, "boundary_diag")
MODEL = "kaggle_bundle_vitb/model_vitb_lora_warm.pt"
PIX, MINA = PT.E17_PIX, PT.E17_MINA
TOL = [1, 2, 4, 8]
MORPH = [-4, -3, -2, -1, 1, 2]                     # negative = erode by that many px
MTHR = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
BINS = [0, 1.01, 2.01, 4.01, 8.01, 1e9]
K3 = np.ones((3, 3), np.uint8)


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else 2.0 * int((a & b).sum()) / s


def run_set(net, ids, device):
    R = {"n": len(ids), "ee": 0, "fp": np.zeros(4, int), "fn": np.zeros(4, int), "tp": np.zeros(4, int),
         "d0": [[] for _ in range(4)], "ratio": [[] for _ in range(4)],
         "fpb": np.zeros((4, 5)), "fnb": np.zeros((4, 5)),
         "tol": {t: [[] for _ in range(4)] for t in TOL},
         "morph": {r: [[] for _ in range(4)] for r in MORPH},
         "mthr": {t: [[] for _ in range(4)] for t in MTHR}}
    for i0 in range(0, len(ids), 4):
        ch = ids[i0:i0 + 4]
        p = PT.member_probs(net, load(os.path.join(DATA_DIR, "train_images"), ch).to(device), PT.TTA, half=True)
        for k, f in enumerate(ch):
            gall = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED)
            for j in range(4):
                full = cv2.resize(p[k, j + 1], (1600, 256), interpolation=cv2.INTER_LINEAR)
                sel = full > PIX[j]
                present = int(sel.sum()) >= MINA[j]
                g = gall == j + 1
                has = bool(g.any())
                if not present and not has:
                    R["ee"] += 1; continue
                if present and not has:
                    R["fp"][j] += 1; continue
                if has and not present:
                    R["fn"][j] += 1; continue
                R["tp"][j] += 1
                d0 = dice(sel, g); R["d0"][j].append(d0); R["ratio"][j].append(sel.sum() / g.sum())
                din = cv2.distanceTransform(g.astype(np.uint8), cv2.DIST_L2, 3)
                dout = cv2.distanceTransform((~g).astype(np.uint8), cv2.DIST_L2, 3)
                dfp, dfn = dout[sel & ~g], din[g & ~sel]
                R["fpb"][j] += np.histogram(dfp, bins=BINS)[0]
                R["fnb"][j] += np.histogram(dfn, bins=BINS)[0]
                I = int((sel & g).sum())
                for t in TOL:
                    far = int((dfp > t).sum() + (dfn > t).sum())
                    R["tol"][t][j].append(2 * I / (2 * I + far) if 2 * I + far else 1.0)
                s8 = sel.astype(np.uint8)
                for r in MORPH:
                    m = cv2.erode(s8, K3, iterations=-r) if r < 0 else cv2.dilate(s8, K3, iterations=r)
                    R["morph"][r][j].append(dice(m.astype(bool), g) - d0)
                for t in MTHR:
                    R["mthr"][t][j].append(dice(full > t, g) - d0)
    return R


def report(name, R):
    P = R["n"] * 4
    tp, fp, fn = R["tp"], R["fp"], R["fn"]
    dsum = np.array([np.sum(R["d0"][j]) for j in range(4)])
    score = (R["ee"] + dsum.sum()) / P
    out = {"images": R["n"], "kaggle": round(score, 5),
           "loss_fp_pairs": round(fp.sum() / P, 5), "loss_fn_pairs": round(fn.sum() / P, 5),
           "loss_mask_on_tp": round((tp.sum() - dsum.sum()) / P, 5),
           "per_class": {}}
    print(f"\n===== {name}: {R['n']} images, Kaggle metric {score:.5f} =====")
    print(f"  lost to FP pairs {fp.sum() / P:.4f} ({fp.sum()}) | FN pairs {fn.sum() / P:.4f} ({fn.sum()}) | "
          f"mask Dice deficit on TP pairs {(tp.sum() - dsum.sum()) / P:.4f} ({tp.sum()} TP pairs)")
    print(f"  {'cls':>4}{'TP':>5}{'FP':>5}{'FN':>5}{'TP Dice':>9}{'area pred/GT':>14}"
          f"{'wrong px: overshoot':>21}{'undershoot':>12}{'<=2px of edge':>15}{'<=8px':>8}")
    for j in range(4):
        if not tp[j]:
            print(f"  {j + 1:>4}{tp[j]:>5}{fp[j]:>5}{fn[j]:>5}"); continue
        ov, un = R["fpb"][j].sum(), R["fnb"][j].sum(); tot = ov + un
        near2 = (R["fpb"][j][:2].sum() + R["fnb"][j][:2].sum()) / tot if tot else 0
        near8 = (R["fpb"][j][:4].sum() + R["fnb"][j][:4].sum()) / tot if tot else 0
        out["per_class"][f"c{j + 1}"] = {"tp": int(tp[j]), "fp": int(fp[j]), "fn": int(fn[j]),
            "tp_dice_mean": round(float(np.mean(R["d0"][j])), 4), "area_ratio_median": round(float(np.median(R["ratio"][j])), 3),
            "overshoot_share": round(ov / tot, 3) if tot else None, "within_2px": round(near2, 3), "within_8px": round(near8, 3)}
        print(f"  {j + 1:>4}{tp[j]:>5}{fp[j]:>5}{fn[j]:>5}{np.mean(R['d0'][j]):>9.3f}{np.median(R['ratio'][j]):>14.2f}"
              f"{100 * ov / tot:>20.0f}%{100 * un / tot:>11.0f}%{100 * near2:>14.0f}%{100 * near8:>7.0f}%")
    print("  boundary-tolerant: metric gain if wrong pixels within k px of the edge were forgiven (an upper bound)")
    out["tolerant_gain"] = {}
    for t in TOL:
        g = sum(np.sum(np.array(R["tol"][t][j]) - np.array(R["d0"][j])) for j in range(4)) / P
        out["tolerant_gain"][t] = round(g, 5)
        print(f"    k={t}px: +{g:.5f}  (TP Dice {np.mean([v for j in range(4) for v in R['tol'][t][j]]):.3f})")
    print("  post-hoc erode(-)/dilate(+) predicted masks, presence fixed: metric change [all | c1 c2 c3 c4]")
    out["morph"] = {}
    for r in MORPH:
        per = [np.sum(R["morph"][r][j]) / P for j in range(4)]
        out["morph"][r] = round(sum(per), 5)
        print(f"    {r:+d}px: {sum(per):+.5f} | " + " ".join(f"{v:+.5f}" for v in per))
    print("  decoupled mask threshold (presence by current rule): metric change [all | c1 c2 c3 c4]")
    out["mask_thr"] = {}
    for t in MTHR:
        per = [np.sum(R["mthr"][t][j]) / P for j in range(4)]
        out["mask_thr"][t] = round(sum(per), 5)
        print(f"    mask pix {t:.2f}: {sum(per):+.5f} | " + " ".join(f"{v:+.5f}" for v in per))
    return out


def main():
    device = "cuda"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    net = PT.load_member(MODEL, device)
    hold = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    tr = pd.read_csv(os.path.join(SPLIT_DIR, "train12k_ids.csv"))["ImageId"].tolist()
    tr = [tr[i] for i in np.sort(np.random.default_rng(0).choice(len(tr), 500, replace=False))]
    print(f"rule pix {PIX} min_area {MINA} | model {MODEL}")
    res = {"model": MODEL, "pix": PIX, "mina": MINA}
    for name, ids in (("TRAIN (seen in training)", tr), ("HOLDOUT (unseen)", hold)):
        res[name.split()[0].lower()] = report(name, run_set(net, ids, device))
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
