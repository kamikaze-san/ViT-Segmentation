"""Why did extra pseudo members LOWER the ensemble's private score?

V79 (1 pseudo member) 0.90874 -> V82 (2) 0.90837 -> V83 (3) 0.90811, while each
pseudo member alone beat its original by +0.0018 to +0.0052. Two explanations
predict different things:

  CORRELATION   members distilled from one teacher on the same 3,402 images make
                the same decisions -> pairwise disagreement drops, averaging adds
                less. Measured label-free on the 2,104 test images that were NOT
                pseudo-labelled (the other 3,402 are memorised, reported separately).
  CALIBRATION   pseudo members trained on thresholded teacher masks are sharper ->
                the averaged probabilities shift and the tuned PIX/MINA no longer
                fit. Measured on holdout (reliable for decision-rule tuning, E12) by
                re-tuning per class for each ensemble with the Kaggle cell's exact
                inference: bundles, 3-view TTA, fp16, bilinear to full resolution.

Trains nothing. Holdout keeps only area/intersection tables, never full
probability maps, so host RAM stays small. Test runs single-view fp32 so the
originals' decisions can be checked against the E17 caches.
"""
import json, os, time
import numpy as np, pandas as pd, cv2, torch

import pseudo_teacher as PT
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR

OUT = os.path.join(PT.OUTPUTS, "pseudo_diag")
MODELS = {
    "vitb":             "kaggle_bundle_vitb/model_vitb.pt",
    "vits_lora_r16":    "kaggle_bundle_lora/model_vits_lora_r16.pt",
    "lora_r8":          "kaggle_bundle_lora/model_lora_r8.pt",
    "bce16":            "kaggle_bundle_bce/model_bce16.pt",
    "vitb_warm_pseudo": "kaggle_bundle_pseudo/model_vitb_lora_warm_pseudo01.pt",
    "vits_r16_pseudo":  "kaggle_bundle_pseudo/model_vits_lora_r16_pseudo01.pt",
    "bce16_pseudo":     "kaggle_bundle_pseudo/model_bce16_pseudo01.pt",
}
E17_NAME = {"vitb": "vitb", "vits_lora_r16": "vits_lora_r16", "lora_r8": "lora_r8", "bce16": "bce16"}
ENSEMBLES = {
    "V79": ["vitb", "vits_lora_r16", "lora_r8", "bce16", "vitb_warm_pseudo"],
    "V82": ["vitb", "vits_lora_r16", "lora_r8", "bce16_pseudo", "vitb_warm_pseudo"],
    "V83": ["vitb", "vits_r16_pseudo", "lora_r8", "bce16_pseudo", "vitb_warm_pseudo"],
}
KAGGLE_PRIVATE = {"V79": 0.90874, "V82": 0.90837, "V83": 0.90811}
TUNED_PIX, TUNED_MINA = [0.65, 0.60, 0.65, 0.58], [630, 1470, 1050, 3150]
PIXG = np.array([0.40, 0.45, 0.50, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75], np.float32)
MINAG = [0, 210, 420, 630, 840, 1050, 1260, 1470, 1750, 2100, 2450, 2800, 3150, 3500, 4200, 5250]
PAIRS = [("vits_lora_r16", "vits_r16_pseudo"), ("bce16", "bce16_pseudo")]   # original -> its pseudo version


def load(dirname, files):
    return torch.from_numpy(np.stack([
        (((cv2.imread(os.path.join(dirname, f))[:, :, ::-1].astype(np.float32) / 255.0)
          - PT.MEAN) / PT.STD).transpose(2, 0, 1) for f in files]))


def class_score(area, inter, g, pi, ma):
    keep = area[:, pi] >= ma
    a = np.where(keep, area[:, pi], 0); i = np.where(keep, inter[:, pi], 0)
    both = (a == 0) & (g == 0); one = (a == 0) ^ (g == 0); ok = ~both & ~one
    return (both.sum() + (2 * i[ok] / (a[ok] + g[ok])).sum()) / len(g)


def main():
    device = "cuda"
    os.makedirs(OUT, exist_ok=True)
    nets = {}
    for m, rel in MODELS.items():
        nets[m] = PT.load_member(rel, device)
    res = {"holdout": {}, "test": {}}

    # ---------------- holdout: threshold fit per ensemble ----------------
    hold = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    n, P = len(hold), len(PIXG)
    area = {e: np.zeros((n, 4, P), np.int64) for e in ENSEMBLES}
    inter = {e: np.zeros((n, 4, P), np.int64) for e in ENSEMBLES}
    gta = np.zeros((n, 4), np.int64)
    t0 = time.time()
    for i0 in range(0, n, 4):
        ch = hold[i0:i0 + 4]
        x = load(os.path.join(DATA_DIR, "train_images"), ch).to(device)
        probs = {m: PT.member_probs(nets[m], x, PT.TTA, half=True) for m in MODELS}
        masks = [cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED) for f in ch]
        for k, g in enumerate(masks):
            for j in range(4):
                gta[i0 + k, j] = (g == j + 1).sum()
        for e, mem in ENSEMBLES.items():
            avg = sum(probs[m] for m in mem) / len(mem)
            for k, g in enumerate(masks):
                for j in range(4):
                    full = cv2.resize(avg[k, j + 1], (1600, 256), interpolation=cv2.INTER_LINEAR)
                    sel = full[None] > PIXG[:, None, None]
                    gc = g == j + 1
                    area[e][i0 + k, j] = sel.sum((1, 2))
                    inter[e][i0 + k, j] = (sel & gc[None]).sum((1, 2))
        if (i0 // 4) % 25 == 0:
            print(f"  holdout {min(i0 + 4, n)}/{n} | {time.time() - t0:.0f}s", flush=True)

    tuned_idx = [int(np.where(np.isclose(PIXG, p))[0][0]) for p in TUNED_PIX]
    print(f"\nHOLDOUT ({n} images, 3-view TTA, fp16, bilinear) - Kaggle metric")
    print(f"{'ens':<5}{'kaggle private':>15}{'@tuned':>10}{'@own best':>11}{'gain':>9}   best per class (pix, min_area)")
    for e in ENSEMBLES:
        at_tuned = [class_score(area[e][:, j], inter[e][:, j], gta[:, j], tuned_idx[j], TUNED_MINA[j]) for j in range(4)]
        best = []
        for j in range(4):
            top = max(((class_score(area[e][:, j], inter[e][:, j], gta[:, j], pi, ma), float(PIXG[pi]), ma)
                       for pi in range(P) for ma in MINAG), key=lambda t: t[0])
            best.append(top)
        s_t, s_b = float(np.mean(at_tuned)), float(np.mean([b[0] for b in best]))
        res["holdout"][e] = {"at_tuned": round(s_t, 5), "at_tuned_per_class": np.round(at_tuned, 4).tolist(),
                             "own_best": round(s_b, 5), "best_pix": [b[1] for b in best],
                             "best_mina": [b[2] for b in best], "best_per_class": [round(b[0], 4) for b in best]}
        print(f"{e:<5}{KAGGLE_PRIVATE[e]:>15.5f}{s_t:>10.5f}{s_b:>11.5f}{s_b - s_t:>+9.5f}   "
              f"pix {[b[1] for b in best]} min_area {[b[2] for b in best]}")
    for e in ENSEMBLES:
        print(f"  {e} per class at tuned {res['holdout'][e]['at_tuned_per_class']} | at own best {res['holdout'][e]['best_per_class']}")

    # ---------------- test: decisions, similarity, rates ----------------
    test = sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))
    nt = len(test)
    pseudo = set(pd.read_csv(os.path.join(SPLIT_DIR, "pseudo_ids.csv"))["ImageId"])
    in_ps = np.array([f in pseudo for f in test])
    dec = {m: np.zeros((nt, 4), bool) for m in MODELS}
    edec = {e: np.zeros((nt, 4), bool) for e in ENSEMBLES}
    t1 = time.time()
    for i0 in range(0, nt, 4):
        sl = slice(i0, i0 + 4)
        x = load(os.path.join(DATA_DIR, "test_images"), test[sl]).to(device)
        probs = {m: PT.member_probs(nets[m], x, ("none",), half=False) for m in MODELS}
        for m in MODELS:
            dec[m][sl] = PT.decide(probs[m], PT.E17_PIX, PT.E17_MINA)[0]
        for e, mem in ENSEMBLES.items():
            edec[e][sl] = PT.decide(sum(probs[m] for m in mem) / len(mem), TUNED_PIX, TUNED_MINA)[0]
        if (i0 // 4) % 250 == 0:
            done = i0 + 4; el = time.time() - t1
            print(f"  test {min(done, nt)}/{nt} | ETA {(nt - done) * el / done / 60:5.1f} min", flush=True)
    for m, name in E17_NAME.items():
        ref = np.load(os.path.join(PT.OUTPUTS, "complement", f"dec_{name}.npy"))
        print(f"  E17 check {m:<14}: {int((dec[m] != ref).sum())} mismatching decisions of {ref.size}")
    np.savez_compressed(os.path.join(OUT, "test_decisions.npz"), **{f"m_{k}": v for k, v in dec.items()},
                        **{f"e_{k}": v for k, v in edec.items()}, in_pseudo=in_ps)

    names = list(MODELS)
    for label, mask in (("NOT pseudo-labelled", ~in_ps), ("pseudo-labelled (memorised)", in_ps)):
        sub = {"n_images": int(mask.sum())}
        dis = lambda a, b: 100 * float((dec[a][mask] != dec[b][mask]).mean())
        print(f"\nTEST, {label}: {int(mask.sum())} images - pairwise disagreement % of image-class decisions")
        print(f"{'':<18}" + "".join(f"{m[:10]:>11}" for m in names))
        for a in names:
            print(f"{a:<18}" + "".join(f"{dis(a, b):>11.2f}" for b in names))
        for e, mem in ENSEMBLES.items():
            pw = [dis(a, b) for ai, a in enumerate(mem) for b in mem[ai + 1:]]
            unan = float((np.stack([dec[m][mask] for m in mem]).all(0) | (~np.stack([dec[m][mask] for m in mem])).all(0)).mean())
            rate = (100 * edec[e][mask].mean(0)).round(2).tolist()
            sub[e] = {"mean_pairwise_disagreement": round(float(np.mean(pw)), 3), "unanimous_pct": round(100 * unan, 2),
                      "ensemble_positive_rate": rate}
            print(f"  {e}: mean pairwise disagreement {np.mean(pw):.3f}% | members unanimous {100 * unan:.2f}% | "
                  f"ensemble positive rate % (tuned thresholds) {rate}")
        sub["pseudo_vs_own_original"] = {b: round(dis(a, b), 3) for a, b in PAIRS}
        sub["between_originals_vits_bce"] = round(dis("vits_lora_r16", "bce16"), 3)
        sub["between_pseudo_vits_bce"] = round(dis("vits_r16_pseudo", "bce16_pseudo"), 3)
        print(f"  vits_lora_r16 vs bce16 (originals): {sub['between_originals_vits_bce']:.3f}% | "
              f"their pseudo versions: {sub['between_pseudo_vits_bce']:.3f}% | "
              f"each pseudo vs own original: {sub['pseudo_vs_own_original']}")
        res["test"][label] = sub
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
