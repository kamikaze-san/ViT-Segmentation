"""Is the mask overshoot (E24) separate spurious blobs, or true regions drawn too big?

Same model, inference and rule as boundary_diag.py (vitb_lora_warm, 3-view TTA,
fp16, bilinear, single-model PIX/MINA), 500 train12k + 500 holdout images.

  A. On true-positive pairs, split wrong pixels:
       overshoot  -> in a predicted component that never touches the label ("spurious")
                     vs outside the label but in a component that does ("extension")
       undershoot -> in a labelled component the prediction never touches ("missed blob")
                     vs a labelled component it only partly covers ("partial")
  B. Component filtering BEFORE the presence decision, as the Kaggle cell would apply
     it: drop predicted components below an absolute area, below a peak probability,
     or below a fraction of the largest component in that image-class. Full Kaggle
     metric, so filtering can also turn a false-alarm pair empty or a true pair empty.
     The unfiltered setting must reproduce boundary_diag's score - a built-in check.

Holdout caveat (E21): it has ~10x fewer false-alarm opportunities than the test set,
so any gain that comes from removing false-alarm pairs is UNDER-counted here.
"""
import json, os, time
import numpy as np, pandas as pd, cv2

import pseudo_teacher as PT
from pseudo_diag import load
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR

OUT = os.path.join(PT.OUTPUTS, "boundary_diag")
MODEL = "kaggle_bundle_vitb/model_vitb_lora_warm.pt"
PIX, MINA = PT.E17_PIX, PT.E17_MINA
AREA = [0, 150, 300, 600, 1000, 1500, 2500]
PEAK = [0.0, 0.70, 0.80, 0.90]
REL = [0.0, 0.02, 0.05, 0.10, 0.20, 0.35]


def components(full, sel, g):
    """Per predicted component: area, peak prob, pixels inside the label."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sel.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros((0, 3))
    area = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    ls, v = lab[sel], full[sel]
    o = np.argsort(ls, kind="stable")
    ls, v = ls[o], v[o]
    starts = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]])
    peak = np.zeros(n - 1)
    peak[ls[starts] - 1] = np.maximum.reduceat(v, starts)
    inter = np.bincount(lab[g], minlength=n)[1:].astype(np.float64)
    return np.stack([area, peak, inter], 1)


def run_set(net, ids, device):
    pairs = []                                           # (class, gt_area, comps)
    dec = {"over_spurious": np.zeros(4), "over_extension": np.zeros(4),
           "under_missed_blob": np.zeros(4), "under_partial": np.zeros(4)}
    for i0 in range(0, len(ids), 4):
        ch = ids[i0:i0 + 4]
        p = PT.member_probs(net, load(os.path.join(DATA_DIR, "train_images"), ch).to(device), PT.TTA, half=True)
        for k, f in enumerate(ch):
            gall = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED)
            for j in range(4):
                full = cv2.resize(p[k, j + 1], (1600, 256), interpolation=cv2.INTER_LINEAR)
                sel = full > PIX[j]
                g = gall == j + 1
                c = components(full, sel, g)
                pairs.append((j, float(g.sum()), c))
                if g.any() and c[:, 0].sum() >= MINA[j]:     # true-positive pair under the current rule
                    dec["over_spurious"][j] += c[c[:, 2] == 0, 0].sum()
                    dec["over_extension"][j] += (c[c[:, 2] > 0, 0] - c[c[:, 2] > 0, 2]).sum()
                    gn, glab = cv2.connectedComponents(g.astype(np.uint8), connectivity=8)
                    cover = np.bincount(glab[sel], minlength=gn)
                    gsize = np.bincount(glab.ravel(), minlength=gn)
                    missed = np.flatnonzero(cover[1:] == 0) + 1
                    dec["under_missed_blob"][j] += gsize[missed].sum()
                    dec["under_partial"][j] += (gsize[1:].sum() - gsize[missed].sum()) - cover[1:].sum()
    return pairs, dec


def score(pairs, amin, pmin, rel):
    """Full Kaggle metric after filtering components, per class sums."""
    tot = np.zeros(4); cnt = np.zeros(4)
    for j, ga, c in pairs:
        cnt[j] += 1
        if len(c):
            keep = (c[:, 0] >= amin) & (c[:, 1] >= pmin) & (c[:, 0] >= rel * c[:, 0].max())
            a, it = c[keep, 0].sum(), c[keep, 2].sum()
        else:
            a = it = 0.0
        present = a >= MINA[j]
        if not present:
            tot[j] += 1.0 if ga == 0 else 0.0
        elif ga > 0:
            tot[j] += 2 * it / (a + ga)
    return tot, cnt


def report(name, pairs, dec, base_check):
    P = len(pairs)
    print(f"\n===== {name} =====")
    over = dec["over_spurious"] + dec["over_extension"]; under = dec["under_missed_blob"] + dec["under_partial"]
    print(f"  wrong pixels on true-positive pairs        c1      c2      c3      c4     all")
    for lbl, num, den in (("overshoot in spurious blobs  (% of over)", dec["over_spurious"], over),
                          ("undershoot in missed blobs  (% of under)", dec["under_missed_blob"], under)):
        tot = 100 * num.sum() / max(den.sum(), 1)
        print(f"  {lbl:<42}" + "".join(f"{100 * a / max(b, 1):>7.0f}%" for a, b in zip(num, den)) + f"{tot:>7.0f}%")
    print(f"  overshoot share of all wrong pixels        " + "".join(
        f"{100 * o / max(o + u, 1):>7.0f}%" for o, u in zip(over, under)) + f"{100 * over.sum() / max(over.sum() + under.sum(), 1):>7.0f}%")

    t0, c0 = score(pairs, 0, 0.0, 0.0)
    base = t0.sum() / P
    print(f"  unfiltered metric {base:.5f} (boundary_diag: {base_check:.5f}) "
          f"{'- reproduces' if abs(base - base_check) < 1e-5 else '- MISMATCH'}")
    rows = []
    for a in AREA:
        for q in PEAK:
            t, _ = score(pairs, a, q, 0.0); rows.append(("area", a, q, 0.0, t))
    for r in REL[1:]:
        t, _ = score(pairs, 0, 0.0, r); rows.append(("rel", 0, 0.0, r, t))
    print(f"  component filtering, metric change vs unfiltered   [all | c1 c2 c3 c4]")
    res = {"unfiltered": round(base, 5), "configs": []}
    for kind, a, q, r, t in rows:
        d = (t - t0) / P
        res["configs"].append({"min_area": a, "min_peak": q, "min_rel": r, "delta": round(float(d.sum()), 5),
                               "per_class": [round(float(v), 5) for v in d]})
        if (kind == "area" and (q == 0.0 or a == 0)) or kind == "rel":
            lab = f"rel >= {r:.2f} of largest" if kind == "rel" else f"area >= {a:>4}, peak >= {q:.2f}"
            print(f"    {lab:<28}{d.sum():+.5f} | " + " ".join(f"{v:+.5f}" for v in d))
    best = max(res["configs"], key=lambda z: z["delta"])
    best_pc = [max(res["configs"], key=lambda z: z["per_class"][j]) for j in range(4)]
    print(f"  best single setting: area {best['min_area']} peak {best['min_peak']} rel {best['min_rel']} -> {best['delta']:+.5f}")
    print("  best per class: " + " | ".join(
        f"c{j + 1} area {b['min_area']} peak {b['min_peak']} rel {b['min_rel']} {b['per_class'][j]:+.5f}" for j, b in enumerate(best_pc)))
    res["decomposition"] = {k: v.tolist() for k, v in dec.items()}
    res["best"] = best; res["best_per_class"] = best_pc
    return res


def main():
    device = "cuda"
    t0 = time.time()
    net = PT.load_member(MODEL, device)
    prev = json.load(open(os.path.join(OUT, "summary.json")))
    hold = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    tr = pd.read_csv(os.path.join(SPLIT_DIR, "train12k_ids.csv"))["ImageId"].tolist()
    tr = [tr[i] for i in np.sort(np.random.default_rng(0).choice(len(tr), 500, replace=False))]
    res = {}
    for key, name, ids in (("train", "TRAIN (seen in training)", tr), ("holdout", "HOLDOUT (unseen)", hold)):
        pairs, dec = run_set(net, ids, device)
        res[key] = report(name, pairs, dec, prev[key]["kaggle"])
    json.dump(res, open(os.path.join(OUT, "components.json"), "w"), indent=1)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
