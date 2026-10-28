"""E30 follow-ups (CPU only).

1 Does a test-like validation predict Kaggle?  10 past single models with cached holdout probability maps
  (outputs/ensemble/probs_<m>_holdout.npy, stride 4) and a Kaggle private score. Each is scored with the Kaggle
  metric on the 500 holdout strips under ONE common decision rule (the E17 single-model rule, bilinear), three ways:
  unweighted (= E12), importance-weighted by the density ratio P(test)/(1-P(test)) from E30 (capped), and on the most
  test-like 30% of holdout strips. Then Pearson/Spearman of each against the Kaggle private score.
2 Where does the ensemble fire on test?  The V68 teacher ensemble's decisions on the 5,506 test strips
  (outputs/pseudo/ens_dec.npy) by darkness bucket, next to the train12k defect rate in the same bucket.
"""
import json, os
import numpy as np, pandas as pd, cv2
from scipy.stats import pearsonr, spearmanr

from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR

ROOT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal"
ENS, SH = os.path.join(ROOT, "outputs", "ensemble"), os.path.join(ROOT, "outputs", "shift")
PIX, MINA = [0.60, 0.60, 0.60, 0.45], [1050, 2450, 1750, 5250]
MODELS = {"12k_ep16": 0.89894, "blackout": 0.89933, "lovasz": 0.89084, "vitb": 0.90006, "bce16": 0.90082,
          "vits_taps": 0.89539, "vitb_lora": 0.89953, "vitl_lora": 0.89956, "vitl_bce": 0.8988, "lora_r8": 0.89961}
SIGMOID = {"bce16", "vitl_bce"}            # bce models: probs cache may hold sigmoid channel probabilities already


def per_image_scores(probs, gts):
    s = np.zeros(len(gts))
    for i in range(len(gts)):
        d = []
        for j, c in enumerate(range(1, 5)):
            pm = cv2.resize(probs[i, c].astype(np.float32), (1600, 256), interpolation=cv2.INTER_LINEAR) > PIX[j]
            if pm.sum() < MINA[j]:
                pm[:] = False
            g = gts[i] == c
            if not pm.any() and not g.any():
                d.append(1.0)
            else:
                d.append(2 * (pm & g).sum() / (pm.sum() + g.sum()))
        s[i] = np.mean(d)
    return s


def main():
    tr = pd.read_csv(os.path.join(SPLIT_DIR, "train12k_ids.csv"))["ImageId"].tolist()
    ho = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    p = np.load(os.path.join(SH, "p_test_oof.npy"))[len(tr):len(tr) + len(ho)]
    w = np.minimum(p / (1 - p), 20.0)
    test_like = p >= np.quantile(p, 0.7)
    ess = w.sum() ** 2 / (w ** 2).sum()
    gts = [cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED) for f in ho]
    rows = []
    for m, kag in MODELS.items():
        pr = np.load(os.path.join(ENS, f"probs_{m}_holdout.npy"), mmap_mode="r")
        s = per_image_scores(pr, gts)
        rows.append({"model": m, "kaggle_private": kag, "holdout_unweighted": s.mean(),
                     "holdout_importance_weighted": float((w * s).sum() / w.sum()), "holdout_test_like_30pct": s[test_like].mean()})
        print(f"{m:10s} kaggle {kag:.5f} | unweighted {s.mean():.4f} | weighted {rows[-1]['holdout_importance_weighted']:.4f} "
              f"| test-like 30% {rows[-1]['holdout_test_like_30pct']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    res = {"decision_rule": {"pix": PIX, "mina": MINA, "bilinear": True}, "weights": "min(p/(1-p), 20)",
           "effective_sample_size_of_weighted_holdout": round(float(ess), 1), "per_model": rows, "correlation_with_kaggle": {}}
    for col in ("holdout_unweighted", "holdout_importance_weighted", "holdout_test_like_30pct"):
        res["correlation_with_kaggle"][col] = {"pearson": round(float(pearsonr(df[col], df.kaggle_private)[0]), 3),
                                               "spearman": round(float(spearmanr(df[col], df.kaggle_private)[0]), 3)}
    print(json.dumps(res["correlation_with_kaggle"], indent=1), "| ESS", round(ess, 1))

    # ---- 2 test firing by darkness bucket vs train defect rate
    te_sorted = sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))
    te_ens = pd.read_csv(os.path.join(ROOT, "outputs", "pseudo", "test_ids.csv")).iloc[:, 0].tolist()
    ens = np.load(os.path.join(ROOT, "outputs", "pseudo", "ens_dec.npy")).astype(bool)
    ens = ens.reshape(len(ens), -1)[:, -4:]
    pos = {f: i for i, f in enumerate(te_ens)}
    ens = ens[[pos[f] for f in te_sorted]]                      # align to the stats order (sorted file names)
    dte = np.load(os.path.join(SH, "stats_test.npy"))[:, 2]
    dtr = np.load(os.path.join(SH, "stats_train12k.npy"))[:, 2]
    ytr = np.array([[(cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED) == c).any()
                     for c in range(1, 5)] for f in tr])
    buckets = [(0, 0.05, "<5% dark"), (0.05, 0.3, "5-30% dark"), (0.3, 0.6, "30-60% dark"), (0.6, 1.01, ">60% dark")]
    res["firing_by_darkness"] = {}
    for lo, hi, name in buckets:
        st, sr = (dte >= lo) & (dte < hi), (dtr >= lo) & (dtr < hi)
        res["firing_by_darkness"][name] = {
            "test_share": round(float(st.mean()), 4), "train_share": round(float(sr.mean()), 4),
            "ensemble_fire_rate_test_c1_c4": [round(float(v), 4) for v in ens[st].mean(0)],
            "train_defect_rate_c1_c4": [round(float(v), 4) for v in ytr[sr].mean(0)],
            "ensemble_fires_any_test": round(float(ens[st].any(1).mean()), 4),
            "train_any_defect": round(float(ytr[sr].any(1).mean()), 4)}
        r = res["firing_by_darkness"][name]
        print(f"{name:12s} test {100*r['test_share']:5.1f}% train {100*r['train_share']:5.1f}% | ens fires any {r['ensemble_fires_any_test']:.3f} "
              f"vs train any-defect {r['train_any_defect']:.3f} | c1-c4 fire {r['ensemble_fire_rate_test_c1_c4']} vs train {r['train_defect_rate_c1_c4']}")
    json.dump(res, open(os.path.join(SH, "shift_followups.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
