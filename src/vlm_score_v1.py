"""Score the zero-shot VLM arms against holdout labels, next to a specialist reference.

Reference: max predicted probability per class from the 12k_ep16 ViT-S member (outputs/ensemble/
gathered_12k_holdout.npz, same 500 holdout images in holdout_ids order) - an older, weaker member than
anything in V79, so it is a conservative specialist baseline.
"""
import json, os
import numpy as np, pandas as pd, cv2

from train_lora import MASK_DIR, SPLIT_DIR

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs"


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    ok = ~np.isnan(s); s, y = s[ok], y[ok]
    if y.all() or not y.any():
        return float("nan")
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort") + 1.0
    for v in np.unique(s):                            # average ranks for ties
        t = s == v
        r[t] = r[t].mean()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)} if os.path.exists(path) else {}


def main():
    ids = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    masks = {f: cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED) for f in ids}
    gt = np.array([[(masks[f] == c).any() for c in range(1, 5)] for f in ids])
    ref = np.load(os.path.join(OUT, "ensemble", "gathered_12k_holdout.npz"))["maxp"][:, 1:5]
    res = {}

    L0 = load(os.path.join(OUT, "vlm", "armA_L0.jsonl"))
    if L0:
        idx = [i for i, f in enumerate(ids) if f in L0 and L0[f].get("p_yes") is not None]
        p = np.array([L0[ids[i]]["p_yes"] for i in idx]); y = gt[idx].any(1)
        hit = tot = clean_box = n_clean = 0
        for i in idx:
            f, boxes = ids[i], L0[ids[i]].get("boxes", [])
            bm = np.zeros((256, 1600), bool)
            for b in boxes:
                x1, y1, x2, y2 = [int(round(v)) for v in (b[0] * 1.6, b[1] * 0.256, b[2] * 1.6, b[3] * 0.256)]
                bm[max(0, y1):max(0, y2) + 1, max(0, x1):max(0, x2) + 1] = True
            if gt[i].any():
                n, lab = cv2.connectedComponents((masks[f] > 0).astype(np.uint8))
                for k in range(1, n):
                    tot += 1; hit += bool((bm & (lab == k)).any())
            else:
                n_clean += 1; clean_box += bool(boxes)
        res["L0"] = {"n": len(idx), "n_defect_images": int(y.sum()),
                     "auc_any_defect": round(auc(p, y), 4),
                     "ref_auc_any_defect": round(auc(ref[idx].max(1), y), 4),
                     "acc_at_0.5": round(float(((p > 0.5) == y).mean()), 4),
                     "says_yes_on_clean": round(float((p[~y] > 0.5).mean()), 4),
                     "says_yes_on_defect": round(float((p[y] > 0.5).mean()), 4),
                     "label_regions_hit_by_a_box": round(hit / max(tot, 1), 4), "label_regions": tot,
                     "clean_images_with_boxes": round(clean_box / max(n_clean, 1), 4)}

    L1 = load(os.path.join(OUT, "vlm", "armA_L1.jsonl"))
    if L1:
        idx = [i for i, f in enumerate(ids) if f in L1 and len(L1[f].get("p_class", {})) == 4]
        P = np.array([[L1[ids[i]]["p_class"][str(c)] for c in range(1, 5)] for i in idx])
        res["L1"] = {"n": len(idx), "unparsed": len([f for f in L1 if len(L1[f].get("p_class", {})) != 4])}
        for c in range(4):
            y = gt[idx, c]
            res["L1"][f"c{c + 1}"] = {"positives": int(y.sum()), "auc": round(auc(P[:, c], y), 4),
                                      "ref_auc": round(auc(ref[idx, c], y), 4),
                                      "yes_rate_on_neg": round(float((P[~y, c] > 0.5).mean()), 4),
                                      "yes_rate_on_pos": round(float((P[y, c] > 0.5).mean()), 4) if y.any() else None}

    C = load(os.path.join(OUT, "vlm", "armC.jsonl"))
    if C:
        r = [v for v in C.values() if "dice_final" in v]
        dpos = [v for v in r if v["label_px"] > 0]; dneg = [v for v in r if v["label_px"] == 0]
        res["C"] = {"n": len(r), "errors": len(C) - len(r),
                    "mean_dice_defect_images_final": round(float(np.mean([v["dice_final"] for v in dpos])), 4) if dpos else None,
                    "mean_dice_defect_images_stage2": round(float(np.mean([v["dice_seg"] for v in dpos])), 4) if dpos else None,
                    "clean_images_left_empty": round(float(np.mean([v["final_px"] == 0 for v in dneg])), 4) if dneg else None,
                    "defect_images_with_any_mask": round(float(np.mean([v["final_px"] > 0 for v in dpos])), 4) if dpos else None,
                    "mean_calls": round(float(np.mean([v["calls"] for v in r])), 2),
                    "mean_sec": round(float(np.mean([v["sec"] for v in r])), 1)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "vlm", "scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
