"""How much do our models use context beyond a 512-px window?

Inference-only diagnostic before any crop training. Each model predicts every
image in these modes:
  full          the normal 256x1600 forward
  window        overlapping 256x512 windows (stride 256 px), stitched so every
                output column comes from the window where it sits furthest from
                an edge (>=128 px of context each side, except at the image border)
  window_nopos  holdout only: the same windows WITHOUT the position fix

The position fix: DINOv3 RoPE normalises patch positions to [-1, 1] over whatever
image it is given (modeling_dinov3_vit.get_patches_center_coordinates), so a bare
512-px window presents 3.1x wider patch spacing than inference ever does. With the
fix, each window gets exactly the positions its patches occupy in the full strip,
so the only thing that differs from `full` is what attention can see.

Reading the result:
  window ~ full     context beyond 512 px carries little information
  window < full     context matters; crops lose information
  window > full     for that class, full-strip predictions lean on context that hurts

Holdout (500 labelled, never trained on by these models): pixel Dice per class
(argmax, as training eval) and the Kaggle metric at the single-model rule.
Test (5,506): empty/non-empty decisions, full vs window. Full-view test decisions
come from outputs/complement (E17); the first --verify-n images are recomputed to
prove this forward pass reproduces them. Models with test-set exposure (the pseudo
student) are deliberately excluded - their full-view test predictions are memorised.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
import transformers.models.dinov3_vit.modeling_dinov3_vit as DM

import pseudo_teacher as PT
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR

OUT = os.path.join(PT.OUTPUTS, "context_diag")
MODELS = {"vitb_lora_warm": "kaggle_bundle_vitb/model_vitb_lora_warm.pt",
          "vits_lora_r16": "kaggle_bundle_lora/model_vits_lora_r16.pt"}
PIX, MINA = PT.E17_PIX, PT.E17_MINA          # single-model rule, same as the E17 decisions
H, W, WIN, STRIDE, PATCH = 256, 1600, 512, 256, 16

# ---- position fix -------------------------------------------------------------
_ORIG_COORDS = DM.get_patches_center_coordinates
_WINDOW = {"col0": None, "full_w": None}


def _windowed_coords(num_patches_h, num_patches_w, dtype=None, device=None):
    if _WINDOW["col0"] is None:
        return _ORIG_COORDS(num_patches_h, num_patches_w, dtype=dtype, device=device)
    fw, c0 = _WINDOW["full_w"], _WINDOW["col0"]
    full = _ORIG_COORDS(num_patches_h, fw, dtype=dtype, device=device).reshape(num_patches_h, fw, 2)
    return full[:, c0:c0 + num_patches_w].reshape(-1, 2)


DM.get_patches_center_coordinates = _windowed_coords   # RoPE forward looks it up at call time

# ---- stitching plan -----------------------------------------------------------
STARTS = list(range(0, W - WIN + 1, STRIDE))
if STARTS[-1] != W - WIN:
    STARTS.append(W - WIN)
CHOICE, MARGIN = np.zeros(W // 4, int), np.zeros(W // 4)
for ci in range(W // 4):
    c = ci * 4 + 2                                   # pixel centre of a stride-4 column
    best = -1.0
    for k, s in enumerate(STARTS):
        if s <= c < s + WIN:
            m = min(np.inf if s == 0 else c - s, np.inf if s + WIN == W else s + WIN - c)
            if m > best:
                best, CHOICE[ci] = m, k
    MARGIN[ci] = best


@torch.no_grad()
def forward(net, x):
    bb, hd, taps, ns, sig = net
    o = bb(pixel_values=x, output_hidden_states=True)
    tok = torch.cat([F.layer_norm(o.hidden_states[t + 1][:, ns:, :],
                                  (o.hidden_states[t + 1].shape[-1],)) for t in taps], -1)
    lg = hd(tok.permute(0, 2, 1).reshape(len(x), -1, x.shape[-2] // PATCH, x.shape[-1] // PATCH)).float()
    if sig:
        p = torch.sigmoid(lg); p[:, 0] = 0.0
        return p
    return lg.softmax(1)


def windowed(net, x, fix=True):
    out = torch.zeros(len(x), 5, H // 4, W // 4, device=x.device)
    for k, s in enumerate(STARTS):
        if fix:
            _WINDOW["col0"], _WINDOW["full_w"] = s // PATCH, W // PATCH
        try:
            p = forward(net, x[..., s:s + WIN])
        finally:
            _WINDOW["col0"] = None
        sel = np.where(CHOICE == k)[0]
        out[..., sel] = p[..., sel - s // 4]
    return out


def load(dirname, files):
    return torch.from_numpy(np.stack([
        (((cv2.imread(os.path.join(dirname, f))[:, :, ::-1].astype(np.float32) / 255.0)
          - PT.MEAN) / PT.STD).transpose(2, 0, 1) for f in files]))


def full_res(p):
    return np.stack([cv2.resize(p[c], (W, H), interpolation=cv2.INTER_LINEAR) for c in range(p.shape[0])])


def decisions(p):
    return np.array([[(cv2.resize(p[k, j + 1], (W, H), interpolation=cv2.INTER_LINEAR) > PIX[j]).sum() >= MINA[j]
                      for j in range(4)] for k in range(len(p))])


def selftest(net, device):
    c = _ORIG_COORDS(16, 100, dtype=torch.float32, device="cpu").reshape(16, 100, 2)
    _WINDOW["col0"], _WINDOW["full_w"] = 17, 100
    got = DM.get_patches_center_coordinates(16, 32, dtype=torch.float32, device="cpu").reshape(16, 32, 2)
    _WINDOW["col0"] = None
    bare = DM.get_patches_center_coordinates(16, 32, dtype=torch.float32, device="cpu").reshape(16, 32, 2)
    assert torch.equal(got, c[:, 17:49]), "patched coords are not the full-strip slice"
    assert not torch.equal(bare, c[:, 17:49]), "unpatched coords unexpectedly equal the slice"
    x = load(os.path.join(DATA_DIR, "test_images"), sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))[:2]).to(device)
    _WINDOW["col0"], _WINDOW["full_w"] = 0, 100
    a = forward(net, x)
    _WINDOW["col0"] = None
    b = forward(net, x)
    assert torch.equal(a, b), "full-width 'window' with the fix does not reproduce the plain forward"
    print(f"  selftest ok: coords slice exact, identity window exact | windows at {STARTS} | "
          f"min interior context {MARGIN[np.isfinite(MARGIN)].min():.0f} px", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--verify-n", type=int, default=64)
    ap.add_argument("--selftest-only", action="store_true")
    a = ap.parse_args()
    device = "cuda"
    os.makedirs(OUT, exist_ok=True)
    hold = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    test = sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))

    for name in a.models:
        print(f"\n### {name}", flush=True)
        net = PT.load_member(MODELS[name], device)
        assert not net[4], "pixel Dice here assumes a softmax model"
        selftest(net, device)
        if a.selftest_only:
            continue
        t0 = time.time()

        # ---- holdout ----
        modes = ("full", "window", "window_nopos")
        inter = {m: np.zeros(5) for m in modes}; psum = {m: np.zeros(5) for m in modes}
        gsum = np.zeros(5); kag = {m: np.zeros(4) for m in modes}
        dec = {m: np.zeros((len(hold), 4), bool) for m in modes}; gt = np.zeros((len(hold), 4), bool)
        for i0 in range(0, len(hold), a.bs):
            ch = hold[i0:i0 + a.bs]
            x = load(os.path.join(DATA_DIR, "train_images"), ch).to(device)
            ps = {"full": forward(net, x), "window": windowed(net, x, True), "window_nopos": windowed(net, x, False)}
            ps = {m: v.cpu().numpy() for m, v in ps.items()}
            for k, f in enumerate(ch):
                g = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED)
                for c in range(1, 5):
                    gsum[c] += (g == c).sum(); gt[i0 + k, c - 1] = (g == c).any()
                for m in modes:
                    fr = full_res(ps[m][k]); am = fr.argmax(0)
                    for c in range(1, 5):
                        pc, gc = am == c, g == c
                        inter[m][c] += (pc & gc).sum(); psum[m][c] += pc.sum()
                        sel = fr[c] > PIX[c - 1]
                        keep = sel.sum() >= MINA[c - 1]
                        dec[m][i0 + k, c - 1] = keep
                        a_, gs = (sel.sum() if keep else 0), gc.sum()
                        kag[m][c - 1] += (1.0 if a_ == 0 and gs == 0 else 0.0 if (a_ == 0 or gs == 0)
                                          else 2 * (sel & gc).sum() / (a_ + gs))
        res = {"holdout": {}, "test": {}}
        print(f"  holdout {len(hold)} images | {time.time() - t0:.0f}s")
        print(f"  {'mode':<14}{'dice c1':>9}{'c2':>8}{'c3':>8}{'c4':>8}{'mean':>8}"
              f"{'kaggle':>9}{'kag c1':>8}{'c2':>7}{'c3':>7}{'c4':>7}{'dec != full':>13}")
        for m in modes:
            d = 2 * inter[m][1:] / np.maximum(psum[m][1:] + gsum[1:], 1)
            k4 = kag[m] / len(hold)
            diff = (dec[m] != dec["full"]).sum(0)
            res["holdout"][m] = {"dice": d.round(4).tolist(), "dice_mean": round(float(d.mean()), 4),
                                 "kaggle": round(float(k4.mean()), 5), "kaggle_per_class": k4.round(4).tolist(),
                                 "decisions_differ_from_full": diff.tolist(),
                                 "decision_errors_vs_gt": (dec[m] != gt).sum(0).tolist()}
            print(f"  {m:<14}" + "".join(f"{v:>8.4f} " if j == 0 else f"{v:>8.4f}" for j, v in enumerate(d))
                  + f"{d.mean():>8.4f}{k4.mean():>9.5f}" + "".join(f"{v:>7.3f}" for v in k4) + f"{str(diff.tolist()):>16}")
        print(f"  decision errors vs GT per class: " + " | ".join(f"{m} {(dec[m] != gt).sum(0).tolist()}" for m in modes))

        # ---- test ----
        ref = np.load(os.path.join(PT.OUTPUTS, "complement", f"dec_{name}.npy"))
        vfull = []
        for i0 in range(0, a.verify_n, a.bs):
            x = load(os.path.join(DATA_DIR, "test_images"), test[i0:i0 + a.bs]).to(device)
            vfull.append(decisions(forward(net, x).cpu().numpy()))
        vfull = np.concatenate(vfull)[:a.verify_n]
        mism = int((vfull != ref[:a.verify_n]).sum())
        print(f"  E17 reproduction on first {a.verify_n} test images: {'exact' if mism == 0 else f'{mism} MISMATCHES'}", flush=True)
        assert mism == 0, "full-view forward does not reproduce E17 - windowed numbers would not be comparable"
        wdec = np.zeros((len(test), 4), bool)
        t1 = time.time()
        for i0 in range(0, len(test), a.bs):
            x = load(os.path.join(DATA_DIR, "test_images"), test[i0:i0 + a.bs]).to(device)
            wdec[i0:i0 + a.bs] = decisions(windowed(net, x, True).cpu().numpy())
            if (i0 // a.bs) % 250 == 0:
                el = time.time() - t1; done = i0 + a.bs
                print(f"    test windowed {min(done, len(test))}/{len(test)} | ETA {(len(test) - done) * el / done / 60:5.1f} min", flush=True)
        np.save(os.path.join(OUT, f"dec_window_{name}.npy"), wdec)
        only_w = (wdec & ~ref).sum(0); only_f = (ref & ~wdec).sum(0)
        res["test"] = {"positive_rate_full": (100 * ref.mean(0)).round(2).tolist(),
                       "positive_rate_window": (100 * wdec.mean(0)).round(2).tolist(),
                       "disagreement_pct": (100 * (wdec != ref).mean(0)).round(2).tolist(),
                       "window_only_positive": only_w.tolist(), "full_only_positive": only_f.tolist()}
        print(f"  TEST 5,506 | positive rate % full {res['test']['positive_rate_full']} "
              f"window {res['test']['positive_rate_window']}")
        print(f"             disagreement % {res['test']['disagreement_pct']} | "
              f"window-only positives {only_w.tolist()} | full-only positives {only_f.tolist()}")
        json.dump(res, open(os.path.join(OUT, f"summary_{name}.json"), "w"), indent=1)
        print(f"  {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
        del net; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
