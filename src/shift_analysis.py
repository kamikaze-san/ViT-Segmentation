"""Train -> test shift analysis (E30). Is the test set distinguishable from training data, by what, and how often do
defects occur in it?

1 features   frozen stock DINOv3 ViT-B, taps 2/5/8/11 LayerNorm'd, pooled mean||max (6,144-d), for all of train12k,
             the 500 holdout strips and the 5,506 competition test strips; plus simple image statistics (brightness,
             contrast, dark/bright fractions, gradient energy, column banding) for interpretation. Cached to
             outputs/shift/.
2 domain     logistic regression "is this a test strip?" on the features, 5-fold out-of-fold over all strips.
             AUC ~0.5 = no appearance shift DINOv3 can see; high AUC = covariate shift. Drivers: which image statistics
             correlate with P(test), and how class prevalence changes across P(test) bins of the labelled strips.
3 prior      per-class presence probabilities from a classifier trained on train12k features, Platt-calibrated on the
             holdout, applied to test; EM prior adjustment (Saerens et al. 2002) estimates the test defect rate per
             class without labels. Compared with the train rate and with the teacher ensemble's firing rate.
"""
import json, os, sys, time
import numpy as np, pandas as pd, cv2, torch
import torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR  # noqa: E402
from encoder_probe import Dino  # noqa: E402
from vlm_score import auc  # noqa: E402

ROOT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal"
OUT = os.path.join(ROOT, "outputs", "shift")


def img_stats(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx, gy = np.abs(np.diff(g, axis=1)).mean(), np.abs(np.diff(g, axis=0)).mean()
    return [g.mean(), g.std(), (g < 20).mean(), (g > 230).mean(), gx, gy, g.mean(0).std(), g.mean(1).std()]


STAT_NAMES = ["brightness", "contrast", "dark_frac", "bright_frac", "grad_x", "grad_y", "column_banding", "row_banding"]


def extract(enc, ids, folder, tag):
    fp = os.path.join(OUT, f"feat_{tag}.npy")
    if os.path.exists(fp):
        return np.load(fp), np.load(os.path.join(OUT, f"stats_{tag}.npy"))
    feats, stats, t0 = [], [], time.time()
    for k, f in enumerate(ids):
        bgr = cv2.imread(os.path.join(DATA_DIR, folder, f))
        feat, _ = enc(bgr[:, :, ::-1].copy())
        feats.append(torch.cat([feat.mean(0), feat.amax(0)]).half().cpu().numpy())
        stats.append(img_stats(bgr))
        if (k + 1) % 2000 == 0:
            print(f"{tag} {k + 1}/{len(ids)} | {(time.time() - t0) / (k + 1):.3f} s/img", flush=True)
    feats, stats = np.stack(feats), np.array(stats, np.float32)
    np.save(fp, feats); np.save(os.path.join(OUT, f"stats_{tag}.npy"), stats)
    return feats, stats


def presence(ids):
    return np.array([[(cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED) == c).any()
                      for c in range(1, 5)] for f in ids])


def fit_logreg(X, Y, wd=1e-2, steps=500, lr=1e-3):
    X, Y = torch.as_tensor(X, dtype=torch.float32).cuda(), torch.as_tensor(Y, dtype=torch.float32).cuda()
    if Y.ndim == 1:
        Y = Y[:, None]
    m = nn.Linear(X.shape[1], Y.shape[1]).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(m(X), Y); opt.zero_grad(); loss.backward(); opt.step()
    return m


def logits(m, X):
    with torch.no_grad():
        return m(torch.as_tensor(X, dtype=torch.float32).cuda()).cpu().numpy()


def em_prior(p, pi_tr, iters=200):
    """Saerens-Latinne-Decaestecker EM for a binary class: adjust calibrated posteriors to an unknown test prior."""
    pi = pi_tr
    for _ in range(iters):
        a = p * (pi / pi_tr); b = (1 - p) * ((1 - pi) / (1 - pi_tr))
        q = a / (a + b)
        pi_new = q.mean()
        if abs(pi_new - pi) < 1e-7:
            break
        pi = pi_new
    return float(pi)


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = pd.read_csv(os.path.join(SPLIT_DIR, "train12k_ids.csv"))["ImageId"].tolist()
    ho = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    # the 5,506 competition test strips = the files in test_images (data/processed/test_ids.csv is the early
    # internal test split of 1,886 TRAINING images, not the competition test set)
    te = sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))
    assert len(te) == 5506, len(te)
    enc = Dino()
    Ftr, Str = extract(enc, tr, "train_images", "train12k")
    Fho, Sho = extract(enc, ho, "train_images", "holdout")
    Fte, Ste = extract(enc, te, "test_images", "test")
    del enc; torch.cuda.empty_cache()
    Ytr, Yho = presence(tr), presence(ho)
    res = {"n": {"train12k": len(tr), "holdout": len(ho), "test": len(te)}}

    # ---- 2 domain classifier, 5-fold out-of-fold over all strips
    X = np.concatenate([Ftr, Fho, Fte]).astype(np.float32); D = np.r_[np.zeros(len(tr) + len(ho)), np.ones(len(te))]
    mu, sd = X.mean(0), X.std(0) + 1e-4; X = (X - mu) / sd
    rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(X)); oof = np.zeros(len(X))
    for k in range(5):
        m = fit_logreg(X[fold != k], D[fold != k])
        oof[fold == k] = 1 / (1 + np.exp(-logits(m, X[fold == k])[:, 0]))
    res["domain_auc_train_vs_test"] = round(auc(oof, D), 4)
    res["domain_auc_holdout_vs_test"] = round(auc(oof[len(tr):], D[len(tr):]), 4)
    S = np.concatenate([Str, Sho, Ste])
    res["image_stats_mean"] = {n: {"train": round(float(Str[:, i].mean()), 4), "holdout": round(float(Sho[:, i].mean()), 4),
                                   "test": round(float(Ste[:, i].mean()), 4)} for i, n in enumerate(STAT_NAMES)}
    res["image_stats_auc_test_vs_train"] = {n: round(auc(S[:, i], D), 4) for i, n in enumerate(STAT_NAMES)}
    lab = np.concatenate([Ytr, Yho]); pl = oof[:len(lab)]
    q = np.quantile(pl, [0, .5, .8, .95, 1])
    bins = {}
    for lo, hi, name in ((q[0], q[1], "least test-like 50%"), (q[1], q[2], "50-80%"), (q[2], q[3], "80-95%"), (q[3], q[4] + 1e-9, "most test-like 5%")):
        sel = (pl >= lo) & (pl < hi)
        bins[name] = {"n": int(sel.sum()), "p_test_range": [round(float(lo), 3), round(float(hi), 3)],
                      "any_defect_rate": round(float(lab[sel].any(1).mean()), 4),
                      "class_rates_c1_c4": [round(float(v), 4) for v in lab[sel].mean(0)]}
    res["labelled_strips_by_test_likeness"] = bins
    res["holdout_p_test_quantiles"] = [round(float(v), 4) for v in np.quantile(oof[len(tr):len(tr) + len(ho)], [.1, .5, .9])]
    np.save(os.path.join(OUT, "p_test_oof.npy"), oof)

    # ---- 3 label-free test prior per class: train presence classifier, Platt on holdout, EM on test
    m = fit_logreg((Ftr.astype(np.float32) - mu) / sd, Ytr)
    zho, zte = logits(m, (Fho.astype(np.float32) - mu) / sd), logits(m, (Fte.astype(np.float32) - mu) / sd)
    pri = {}
    ens = np.load(os.path.join(ROOT, "outputs", "pseudo", "ens_dec.npy")).astype(bool)
    ens = ens.reshape(len(ens), -1)[:, -4:]
    for c in range(4):
        a = torch.zeros(2, requires_grad=True)
        opt = torch.optim.LBFGS([a], max_iter=200)
        zt, yt = torch.tensor(zho[:, c]), torch.tensor(Yho[:, c], dtype=torch.float32)
        def closure():
            opt.zero_grad(); l = F.binary_cross_entropy_with_logits(a[0] * zt + a[1], yt); l.backward(); return l
        opt.step(closure)
        pho = torch.sigmoid(a[0] * zt + a[1]).detach().numpy()
        pte = torch.sigmoid(a[0] * torch.tensor(zte[:, c]) + a[1]).detach().numpy()
        pi_tr = float(Yho[:, c].mean())          # calibration is on the holdout, so its rate is the reference prior
        pri[f"c{c + 1}"] = {"train12k_rate": round(float(Ytr[:, c].mean()), 4), "holdout_rate": round(pi_tr, 4),
                            "holdout_auc": round(auc(pho, Yho[:, c]), 4),
                            "test_mean_calibrated_p": round(float(pte.mean()), 4),
                            "test_rate_em": round(em_prior(pte, pi_tr), 4),
                            "teacher_ensemble_fire_rate_test": round(float(ens[:, c].mean()), 4)}
    res["test_prior_per_class"] = pri
    json.dump(res, open(os.path.join(OUT, "shift_analysis.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
