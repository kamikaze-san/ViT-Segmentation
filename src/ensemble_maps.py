"""Per-member test probability maps, saved once, so any ensemble of them can be rate-matched without re-running models.

  maps   one GPU pass over the 5,506 competition test images: each member's 3-view-TTA probabilities at the head's
         stride-4 resolution (classes 1-4, 64 x 400, fp16), computed as the Kaggle cell does (fp16 autocast; softmax, or
         sigmoid with no background for BCE members), into outputs/ensemble_maps/<member>.npy (1.1 GB each). Members
         already on disk are skipped, so a new model (Qwen, ...) costs only its own pass.
  sets   CPU: average the chosen members' maps (uniform weights, as the cell), bilinear to 256x1600, per-image area at
         every threshold 0.00-0.99; then per class the PIX matching V79's firing rate at its tuned rule (PIX
         [.65,.60,.65,.58], min area [630,1470,1050,3150]). V79 rebuilt from the saved maps must reproduce E26's
         recorded V79 rates (outputs/soft_member/test_areas.npz) - the check that the maps are right.
  python ensemble_maps.py maps [--members vitb vits_r16 lora_r8 bce16 pseudo01 cradio]
  python ensemble_maps.py sets
  python ensemble_maps.py diversity --members vitb vits_r16 lora_r8 bce16 pseudo01 cradio qwen
"""
import argparse, json, os, sys, time
import numpy as np, cv2, torch

import pseudo_teacher as PT
from train_lora import DATA_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "ensemble_maps")
BUNDLES = {"vitb": "kaggle_bundle_vitb/model_vitb.pt",
           "vits_r16": "kaggle_bundle_lora/model_vits_lora_r16.pt",
           "lora_r8": "kaggle_bundle_lora/model_lora_r8.pt",
           "bce16": "kaggle_bundle_bce/model_bce16.pt",
           "pseudo01": "kaggle_bundle_pseudo/model_vitb_lora_warm_pseudo01.pt",
           "soft01": "kaggle_bundle_calib/model_vitb_lora_warm_soft_pseudo01.pt",
           "cradio": "kaggle_bundle_cradio/model_cradio_so_soft_pseudo01.pt",
           "qwen": "kaggle_bundle_qwen/model_qwenvit_soft_pseudo01.pt"}
ADAPTERS = {"cradio": ("cradio", "C-RADIOv4-SO400M"), "qwen": ("qwenvit", "WeVisOne-VL-2B")}   # member -> (kind, external/ folder)
V79 = ["vitb", "vits_r16", "lora_r8", "bce16", "pseudo01"]
SETS = {"V79": V79,
        "V79_add_cradio": V79 + ["cradio"],
        "V79_swap_pseudo01_for_cradio": ["vitb", "vits_r16", "lora_r8", "bce16", "cradio"],
        "V79_swap_lora_r8_for_cradio": ["vitb", "vits_r16", "bce16", "pseudo01", "cradio"],
        "V79_add_qwen": V79 + ["qwen"],
        "V79_add_cradio_qwen": V79 + ["cradio", "qwen"],
        "V79_swap_lora_r8_add_cradio_qwen": ["vitb", "vits_r16", "bce16", "pseudo01", "cradio", "qwen"],
        "V79_swap_pseudo01_add_cradio_qwen": ["vitb", "vits_r16", "lora_r8", "bce16", "cradio", "qwen"]}
ENS_PIX, ENS_MINA = [0.65, 0.60, 0.65, 0.58], [630, 1470, 1050, 3150]
EDGES = np.round(np.arange(0.0, 1.0001, 0.01), 2)
GRID = EDGES[:-1]
H, W = 64, 400


def test_files():
    return sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))


def load_adapter(m, dev):
    """A non-DINOv3 member (backbones.py adapter) from its Kaggle bundle, with its own input normalisation."""
    from backbones import build_backbone
    from spatial_head import build
    kind, folder = ADAPTERS[m]; rel = BUNDLES[m]
    ck = torch.load(os.path.join(PT.OUTPUTS, rel), map_location="cpu", weights_only=False, mmap=True)
    assert ck.get("backbone_kind") == kind, (rel, ck.get("backbone_kind"))
    ad = build_backbone(kind, os.path.join(HERE, "..", "external", folder), dev)
    ad.module.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["backbone"].items()}, strict=True)
    ad.module.eval()
    hd = build(ck.get("head_arch", "conv"), dim=ad.dim * len(ad.taps), upscale=4, hidden=ck["hidden"])
    hd.load_state_dict({k: v.float() for k, v in ck["head"].items()})
    print(f"  {os.path.basename(rel):<26} {kind} taps {ad.taps}", flush=True)
    mean = torch.tensor(ad.norm["mean"], dtype=torch.float32).view(1, 3, 1, 1).to(dev)
    std = torch.tensor(ad.norm["std"], dtype=torch.float32).view(1, 3, 1, 1).to(dev)
    return ad, hd.to(dev).eval(), mean, std


@torch.no_grad()
def adapter_probs(net, x01):
    ad, hd, mean, std = net
    x = (x01 - mean) / std
    acc = 0.0
    for v in PT.TTA:
        xv = x if v == "none" else torch.flip(x, [3] if v == "h" else [2])
        with torch.autocast("cuda", dtype=torch.float16):
            p = hd(ad(xv)).float().softmax(1)
        acc = acc + (p if v == "none" else torch.flip(p, [3] if v == "h" else [2]))
    return (acc / len(PT.TTA)).cpu().numpy()


def run_maps(members, limit=0):
    dev = "cuda"
    out = OUT + "_smoke" if limit else OUT
    os.makedirs(out, exist_ok=True)
    files = test_files()[:limit or None]; n = len(files)
    todo = [m for m in members if not os.path.exists(os.path.join(out, f"{m}.npy"))]
    print(f"members on disk: {[m for m in members if m not in todo]} | to compute: {todo}", flush=True)
    if not todo:
        return
    nets = {m: (load_adapter(m, dev) if m in ADAPTERS else PT.load_member(BUNDLES[m], dev)) for m in todo}
    part = {m: np.lib.format.open_memmap(os.path.join(out, f"{m}.npy.part"), mode="w+", dtype=np.float16,
                                         shape=(n, 4, H, W)) for m in todo}
    mean = torch.tensor(PT.MEAN).view(1, 3, 1, 1).to(dev); std = torch.tensor(PT.STD).view(1, 3, 1, 1).to(dev)
    t0 = time.time()
    for i0 in range(0, n, 4):
        ch = files[i0:i0 + 4]
        x01 = torch.from_numpy(np.stack([(cv2.imread(os.path.join(DATA_DIR, "test_images", f))[:, :, ::-1]
                                          .astype(np.float32) / 255.0).transpose(2, 0, 1) for f in ch])).to(dev)
        xin = (x01 - mean) / std
        for m, net in nets.items():
            p = adapter_probs(net, x01) if m in ADAPTERS else PT.member_probs(net, xin, PT.TTA, half=True)
            part[m][i0:i0 + len(ch)] = p[:, 1:5].astype(np.float16)
        if (i0 // 4) % 250 == 0:
            done = min(i0 + 4, n); el = time.time() - t0
            print(f"  {done}/{n} | {el / done:.3f} s/img | ETA {(n - done) * el / done / 60:5.1f} min", flush=True)
    for m in todo:
        part[m].flush(); del part[m]
        os.replace(os.path.join(out, f"{m}.npy.part"), os.path.join(out, f"{m}.npy"))
    json.dump({"files": files, "members": {m: BUNDLES[m] for m in members}}, open(os.path.join(out, "index.json"), "w"), indent=1)
    print(f"maps done in {(time.time() - t0) / 60:.1f} min", flush=True)


def areas_ge(p_full):
    counts, _ = np.histogram(p_full, bins=EDGES)
    return np.cumsum(counts[::-1])[::-1]


def set_areas(members, maps):
    n = maps[members[0]].shape[0]
    A = np.zeros((n, 4, len(GRID)), np.int32)
    for i in range(n):
        avg = sum(maps[m][i].astype(np.float32) for m in members) / len(members)
        for j in range(4):
            A[i, j] = areas_ge(cv2.resize(avg[j], (1600, 256), interpolation=cv2.INTER_LINEAR))
    return A


def run_sets():
    need = sorted({m for s in SETS.values() for m in s})
    maps = {m: np.load(os.path.join(OUT, f"{m}.npy"), mmap_mode="r") for m in need}
    idx = lambda p: int(round(p * 100))
    rate = lambda A, j, k: float((A[:, j, k] >= ENS_MINA[j]).mean())
    areas, res = {}, {}
    for s, mem in SETS.items():
        t0 = time.time(); areas[s] = set_areas(mem, maps)
        print(f"  {s}: areas in {time.time() - t0:.0f}s", flush=True)
    ref = np.load(os.path.join(HERE, "..", "outputs", "soft_member", "test_areas.npz"))["V79"]
    tgt = [rate(areas["V79"], j, idx(ENS_PIX[j])) for j in range(4)]
    e26 = [rate(ref, j, idx(ENS_PIX[j])) for j in range(4)]
    res["check_V79_vs_E26"] = {"rebuilt_rate_pct": [round(100 * v, 3) for v in tgt], "e26_rate_pct": [round(100 * v, 3) for v in e26],
                               "max_abs_diff_pp": round(100 * max(abs(a - b) for a, b in zip(tgt, e26)), 3)}
    for s, A in areas.items():
        if s == "V79":
            continue
        best = [min(range(len(GRID)), key=lambda k: abs(rate(A, j, k) - tgt[j])) for j in range(4)]
        dec_new = np.stack([A[:, j, best[j]] >= ENS_MINA[j] for j in range(4)], 1)
        dec_v79 = np.stack([areas["V79"][:, j, idx(ENS_PIX[j])] >= ENS_MINA[j] for j in range(4)], 1)
        res[s] = {"members": SETS[s], "reference": "V79 at its tuned rule", "reference_pix": ENS_PIX, "min_area": ENS_MINA,
                  "reference_rate_pct": [round(100 * v, 3) for v in tgt],
                  "rate_at_reference_pix_pct": [round(100 * rate(A, j, idx(ENS_PIX[j])), 3) for j in range(4)],
                  "matched_pix": [float(GRID[b]) for b in best],
                  "matched_rate_pct": [round(100 * rate(A, j, best[j]), 3) for j in range(4)],
                  "pairs_deciding_differently_from_V79_pct": [round(100 * float((dec_new[:, j] != dec_v79[:, j]).mean()), 3) for j in range(4)]}
    np.savez_compressed(os.path.join(OUT, "set_areas.npz"), grid=GRID, **areas)
    json.dump(res, open(os.path.join(OUT, "rate_match.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def run_diversity(members):
    """Each member ALONE, rate-matched to V79's test firing rates (so calibration differences do not count), then:
    % of (image, class) pairs where it decides differently from V79, and pairwise between members."""
    maps = {m: np.load(os.path.join(OUT, f"{m}.npy"), mmap_mode="r") for m in sorted(set(members) | set(V79))}
    idx = lambda p: int(round(p * 100))
    rate = lambda A, j, k: float((A[:, j, k] >= ENS_MINA[j]).mean())
    v79 = set_areas(V79, maps)
    tgt = [rate(v79, j, idx(ENS_PIX[j])) for j in range(4)]
    dec_v79 = np.stack([v79[:, j, idx(ENS_PIX[j])] >= ENS_MINA[j] for j in range(4)], 1)
    dec, res = {}, {"members": {}, "pairwise_pct_mean_over_classes": {}}
    for m in members:
        A = set_areas([m], maps)
        best = [min(range(len(GRID)), key=lambda k: abs(rate(A, j, k) - tgt[j])) for j in range(4)]
        dec[m] = np.stack([A[:, j, best[j]] >= ENS_MINA[j] for j in range(4)], 1)
        res["members"][m] = {"matched_pix_alone": [float(GRID[b]) for b in best],
                             "differs_from_V79_pct": [round(100 * float((dec[m][:, j] != dec_v79[:, j]).mean()), 3) for j in range(4)]}
        print(f"  {m:9s} alone, rate-matched: differs from V79 on {res['members'][m]['differs_from_V79_pct']} % of pairs (c1-c4)", flush=True)
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            res["pairwise_pct_mean_over_classes"][f"{a}~{b}"] = round(100 * float((dec[a] != dec[b]).mean()), 3)
    json.dump(res, open(os.path.join(OUT, "diversity.json"), "w"), indent=1)
    print(json.dumps(res["pairwise_pct_mean_over_classes"], indent=1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["maps", "sets", "diversity"])
    ap.add_argument("--members", nargs="+", default=V79 + ["cradio"])
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N test images, into ensemble_maps_smoke/")
    a = ap.parse_args()
    if a.cmd == "maps":
        run_maps(a.members, a.limit)
    elif a.cmd == "sets":
        run_sets()
    else:
        run_diversity(a.members)
