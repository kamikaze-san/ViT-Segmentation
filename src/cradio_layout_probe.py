"""E31 follow-up, job 1: which layout of C-RADIO's position grid suits our 256x1600 strips?

C-RADIOv4 learns one square 128x128 position grid. At eval time a non-square input gets the grid resized to
max(h, w) x max(h, w) and its top-left h x w window (vit_patch_generator.py), so a strip is treated as the TOP BAND of a
1600x1600 image. Pretraining only ever cropped the grid at aspect 3/4-4/3, so every layout of a 1:6.25 strip is
off-distribution; this measures which one the frozen features like best. Layouts (16 x 100 patch grid):
  top      the default: grid resized to 100x100, rows 0-15
  center   grid resized to 100x100, rows 42-57 (the strip as the middle band of a square image)
  stretch  grid resized straight to 16x100 (anisotropic: rows 6.25x further apart than columns)
  native   no resize: rows 0-15, cols 0-99 of the 128x128 grid (spacing of a 2048-px image)

Protocol = E29's (encoder_probe.py), the same strips and background draws for every layout (paired):
  probe     linear patch classifier on 800 train12k strips (all defect cells + 150 background cells each), soft
            targets at stride 16, per-class Dice on all 500 holdout strips; 3 probe seeds per layout
  presence  per-class logistic regression on pooled [mean || max] features of 3,000 train12k strips, AUC on holdout
  noise     paired bootstrap over holdout strips (2,000 resamples) of each layout's Dice minus top's
Decision: switch only if a layout beats top on mean Dice by >= 0.01 with a 95% interval above 0, since switching
needs a new stage-1 head (the current one was trained on top).

Frozen backbone, no training of it. Memory: ~2 GB of probe features per layout in RAM (~9 GB total: Mac yes, PC no).
  python cradio_layout_probe.py [--backbone-path ../external/C-RADIOv4-SO400M] [--precision auto|bf16|fp32]
"""
import argparse, contextlib, json, os, time, types
import numpy as np, pandas as pd, cv2, torch
import torch.nn as nn, torch.nn.functional as F

from spatial_head import soft_targets_at_stride
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR
from backbones import build_backbone

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "radio_layout_probe")
N_PROBE, N_POOL, BG_PER_IMG, PROBE_SEEDS, N_BOOT = 800, 3000, 150, (0, 1, 2), 2000
LAYOUTS = ("top", "center", "stretch", "native")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def layout_fn(name):
    """Replacement for ViTPatchGenerator._get_pos_embeddings (eval branch) laying the grid out as `name`."""
    def _get(self, batch_size, input_dims):
        pe = self.pos_embed.reshape(1, self.num_rows, self.num_cols, -1).permute(0, 3, 1, 2)
        h, w = input_dims
        if name in ("top", "center"):
            m = max(h, w)
            pe = F.interpolate(pe.float(), size=(m, m), align_corners=False, mode="bilinear").to(pe.dtype)
            r0, c0 = (0, 0) if name == "top" else ((m - h) // 2, (m - w) // 2)
            pe = pe[..., r0:r0 + h, c0:c0 + w]
        elif name == "stretch":
            pe = F.interpolate(pe.float(), size=(h, w), align_corners=False, mode="bilinear").to(pe.dtype)
        elif name == "native":
            assert h <= self.num_rows and w <= self.num_cols, f"{input_dims} exceeds the {self.num_rows} grid"
            pe = pe[..., :h, :w]
        return pe.flatten(2).permute(0, 2, 1)
    return _get


def set_layout(ad, name):
    if name is None:                                   # stock code path
        ad.pos_gen.__dict__.pop("_get_pos_embeddings", None)
    else:
        ad.pos_gen._get_pos_embeddings = types.MethodType(layout_fn(name), ad.pos_gen)


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone-path", default=os.environ.get("CRADIO_PATH", os.path.join(HERE, "..", "external", "C-RADIOv4-SO400M")))
    ap.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp32"],
                    help="auto = bf16 autocast on CUDA (as E29), fp32 elsewhere")
    ap.add_argument("--limit-train", type=int, default=None, help="smoke test: fewer train strips")
    ap.add_argument("--limit-holdout", type=int, default=None, help="smoke test: fewer holdout strips")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = pick_device()
    bf16 = a.precision == "bf16" or (a.precision == "auto" and dev.type == "cuda")
    amp = (lambda: torch.autocast(device_type=dev.type, dtype=torch.bfloat16)) if bf16 else contextlib.nullcontext
    ad = build_backbone("cradio", a.backbone_path, dev)
    ad.module.eval()
    print(f"device {dev} | {'bf16 autocast' if bf16 else 'fp32'} | taps {ad.taps} | grid "
          f"{ad.pos_gen.num_rows}x{ad.pos_gen.num_cols} | layouts {LAYOUTS}", flush=True)

    @torch.no_grad()
    def feats(rgb):
        x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).to(dev)
        with amp():
            out = ad(x)                                # (1, 4*dim, 16, 100), LayerNorm'd per tap as the head reads it
        return out[0].float().flatten(1).T             # (1600, 4*dim)

    tr = pd.read_csv(os.path.join(SPLIT_DIR, "train12k_ids.csv"))["ImageId"].tolist()
    np.random.default_rng(0).shuffle(tr); tr = tr[:a.limit_train or N_POOL]
    ho = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()[:a.limit_holdout or None]
    rd = lambda f: cv2.imread(os.path.join(DATA_DIR, "train_images", f))[:, :, ::-1].copy()
    mask_of = lambda f: cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED)

    # sanity: our "top" must reproduce the stock eval layout exactly
    rgb0 = rd(tr[0])
    set_layout(ad, None); ref = feats(rgb0)
    set_layout(ad, "top"); top = feats(rgb0)
    diff = (ref - top).abs().max().item()
    print(f"layout 'top' vs stock code: max |diff| {diff:.2e}", flush=True)
    assert diff < 1e-3, "the 'top' layout does not reproduce C-RADIO's own eval layout"

    rng = np.random.default_rng(0)
    X = {L: [] for L in LAYOUTS}; Y = []; pool_tr = {L: [] for L in LAYOUTS}; ytr = []
    t0 = time.time()
    for k, f in enumerate(tr):
        rgb, m = rd(f), mask_of(f)
        keep = None
        if k < N_PROBE:
            t = soft_targets_at_stride(m, 16).reshape(5, -1).T
            bg = np.flatnonzero(t[:, 0] >= 0.999)
            keep = np.concatenate([np.flatnonzero(t[:, 0] < 0.999), rng.choice(bg, min(BG_PER_IMG, len(bg)), replace=False)])
            Y.append(torch.from_numpy(t[keep]).float())
        for L in LAYOUTS:
            set_layout(ad, L); ft = feats(rgb)
            pool_tr[L].append(torch.cat([ft.mean(0), ft.amax(0)]).half().cpu())
            if keep is not None:
                X[L].append(ft[torch.from_numpy(keep).to(dev)].half().cpu())
        ytr.append([(m == c).any() for c in range(1, 5)])
        if (k + 1) % 250 == 0:
            print(f"train {k + 1}/{len(tr)} | {(time.time() - t0) / (k + 1):.2f} s/strip (all layouts)", flush=True)
    Y = torch.cat(Y); X = {L: torch.cat(v) for L, v in X.items()}
    D = X[LAYOUTS[0]].shape[1]
    print(f"probe set {tuple(X[LAYOUTS[0]].shape)} per layout | feature dim {D}", flush=True)

    probes = {}
    for L in LAYOUTS:
        for s in PROBE_SEEDS:
            torch.manual_seed(s)
            lin = nn.Linear(D, 5).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
            for _ in range(600):
                j = torch.randint(0, len(X[L]), (8192,))
                loss = F.cross_entropy(lin(X[L][j].to(dev).float()), Y[j].to(dev))
                opt.zero_grad(); loss.backward(); opt.step()
            probes[(L, s)] = lin.eval()
    del X

    ytr_t = torch.tensor(np.array(ytr), dtype=torch.float32).to(dev)
    pres = {}
    for L in LAYOUTS:
        P = torch.stack(pool_tr[L]).float(); mu, sd = P.mean(0), P.std(0) + 1e-4
        torch.manual_seed(0)
        clf = nn.Linear(P.shape[1], 4).to(dev); opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-2)
        Pn = ((P - mu) / sd).to(dev)
        for _ in range(400):
            loss = F.binary_cross_entropy_with_logits(clf(Pn), ytr_t); opt.zero_grad(); loss.backward(); opt.step()
        pres[L] = (clf.eval(), mu.to(dev), sd.to(dev))
    del pool_tr

    # holdout: per-strip counts so layouts can be compared paired, strip by strip
    cnt = {(L, s): np.zeros((len(ho), 3, 5)) for L in LAYOUTS for s in PROBE_SEEDS}   # inter, pred, gt
    sc = {L: [] for L in LAYOUTS}; yho = []
    for i, f in enumerate(ho):
        rgb, m = rd(f), mask_of(f)
        g = soft_targets_at_stride(m, 16).reshape(5, -1).argmax(0)
        yho.append([(m == c).any() for c in range(1, 5)])
        for L in LAYOUTS:
            set_layout(ad, L); ft = feats(rgb)
            with torch.no_grad():
                for s in PROBE_SEEDS:
                    pred = probes[(L, s)](ft).argmax(-1).cpu().numpy()
                    for c in range(1, 5):
                        cnt[(L, s)][i, :, c] = (((pred == c) & (g == c)).sum(), (pred == c).sum(), (g == c).sum())
                clf, mu, sd = pres[L]
                p = (torch.cat([ft.mean(0), ft.amax(0)]) - mu) / sd
                sc[L].append(torch.sigmoid(clf(p[None]))[0].cpu().numpy())
    yho = np.array(yho)

    def dice(c3d, idx=None):                          # c3d: (n, 3, 5) summed over strips (and seeds)
        t = c3d if idx is None else c3d[idx]
        t = t.sum(0)
        return np.array([2 * t[0, c] / max(t[1, c] + t[2, c], 1) for c in range(1, 5)])

    pooled = {L: sum(cnt[(L, s)] for s in PROBE_SEEDS) for L in LAYOUTS}
    brng = np.random.default_rng(0)
    boots = [brng.integers(0, len(ho), len(ho)) for _ in range(N_BOOT)]
    res = {"backbone_path": a.backbone_path, "device": str(dev), "precision": "bf16" if bf16 else "fp32",
           "n_probe_train": min(N_PROBE, len(tr)), "n_pool_train": len(tr), "n_holdout": len(ho),
           "probe_seeds": list(PROBE_SEEDS), "holdout_positives_c1_c4": yho.sum(0).tolist(), "layouts": {}}
    base = dice(pooled["top"]); base_b = np.array([dice(pooled["top"], b) for b in boots])
    for L in LAYOUTS:
        per_seed = np.array([dice(cnt[(L, s)]) for s in PROBE_SEEDS])
        d = dice(pooled[L])
        S = np.array(sc[L])
        r = {"probe_dice_c1_c4": d.round(4).tolist(), "probe_mean_dice": round(float(d.mean()), 4),
             "probe_mean_dice_per_seed": per_seed.mean(1).round(4).tolist(),
             "presence_auc_c1_c4": [round(auc(S[:, c], yho[:, c]), 4) for c in range(4)],
             "presence_auc_any": round(auc(S.max(1), yho.any(1)), 4)}
        if L != "top":
            db = np.array([dice(pooled[L], b) for b in boots]) - base_b
            lo, hi = np.percentile(db.mean(1), [2.5, 97.5])
            lo3, hi3 = np.percentile(db[:, [0, 2, 3]].mean(1), [2.5, 97.5])
            r["vs_top"] = {"mean_dice_diff": round(float(d.mean() - base.mean()), 4),
                           "ci95_mean": [round(float(lo), 4), round(float(hi), 4)],
                           "mean_excl_c2_diff": round(float(d[[0, 2, 3]].mean() - base[[0, 2, 3]].mean()), 4),
                           "ci95_mean_excl_c2": [round(float(lo3), 4), round(float(hi3), 4)],
                           "per_class_diff": (d - base).round(4).tolist(),
                           "better": bool(d.mean() - base.mean() >= 0.01 and lo > 0)}
        res["layouts"][L] = r
    res["minutes"] = round((time.time() - t0) / 60, 1)
    winners = [L for L, r in res["layouts"].items() if r.get("vs_top", {}).get("better")]
    res["verdict"] = (f"switch candidate(s): {winners} (needs a new stage-1 head)" if winners else
                      "keep 'top' (the default): no layout beats it by >= 0.01 with a 95% interval above 0")
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(f"\n{'layout':8s} {'Dice c1 / c2 / c3 / c4':28s} {'mean':>6s} {'vs top [95% CI]':>24s} {'any AUC':>8s}")
    for L, r in res["layouts"].items():
        v = r.get("vs_top")
        vs = f"{v['mean_dice_diff']:+.4f} [{v['ci95_mean'][0]:+.3f}, {v['ci95_mean'][1]:+.3f}]" if v else "-"
        print(f"{L:8s} {' / '.join(f'{x:.3f}' for x in r['probe_dice_c1_c4']):28s} {r['probe_mean_dice']:6.4f} {vs:>24s} "
              f"{r['presence_auc_any']:8.4f}")
    print(f"\n{res['verdict']} | {res['minutes']} min | {os.path.join(OUT, 'results.json')}", flush=True)


if __name__ == "__main__":
    main()
