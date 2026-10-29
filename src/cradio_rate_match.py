"""Solo thresholds for the C-RADIO stage-2 model (E31 run 2), matched to the same test firing rates as the soft ViT-B
solo (E26), label-free. Same procedure as soft_member_match.py:

One pass over all 5,506 competition test images with the Kaggle bundle, computed the way kaggle_ensemble_cell_v2.py
does (fp16-stored weights loaded as fp32, fp16 autocast, 3-view TTA averaged as softmax probabilities, bilinear to
256x1600), records per image and class the area at every probability threshold 0.00-0.99. Then per class the PIX whose
firing rate (area >= min_area) matches the reference: vitb_lora_warm_pseudo01 solo at PIX [.60,.60,.60,.45], min_area
[1050,2450,1750,5250], the same target the soft ViT-B solo (0.90695) was matched to. The reference areas are reused from
outputs/soft_member/test_areas.npz.
  python cradio_rate_match.py [--bundle ../outputs/kaggle_bundle_cradio/model_cradio_so_soft_pseudo01.pt]
"""
import argparse, json, os, time
import numpy as np, cv2, torch

from train_lora import DATA_DIR
from backbones import build_backbone
from spatial_head import build

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "cradio_rate_match")
REF_NPZ = os.path.join(HERE, "..", "outputs", "soft_member", "test_areas.npz")
SOLO_PIX, SOLO_MINA = [0.60, 0.60, 0.60, 0.45], [1050, 2450, 1750, 5250]
SOFT01_PIX = [0.51, 0.43, 0.44, 0.32]                     # the soft ViT-B solo's matched PIX (E26), as a cross-check
TTA = ("none", "h", "v")
EDGES = np.round(np.arange(0.0, 1.0001, 0.01), 2)
GRID = EDGES[:-1]


def areas_ge(p_full):
    counts, _ = np.histogram(p_full, bins=EDGES)
    return np.cumsum(counts[::-1])[::-1]                   # pixels with prob >= GRID[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(HERE, "..", "outputs", "kaggle_bundle_cradio", "model_cradio_so_soft_pseudo01.pt"))
    ap.add_argument("--backbone-path", default=os.path.join(HERE, "..", "external", "C-RADIOv4-SO400M"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"
    ck = torch.load(a.bundle, map_location="cpu", weights_only=False, mmap=True)
    assert ck.get("backbone_kind") == "cradio", ck.get("backbone_kind")
    ad = build_backbone("cradio", a.backbone_path, dev)
    ad.module.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["backbone"].items()}, strict=True)
    ad.module.eval()
    assert tuple(ck["tap_layers"]) == tuple(ad.taps), (ck["tap_layers"], ad.taps)
    hd = build(ck.get("head_arch", "conv"), dim=ad.dim * len(ad.taps), upscale=4, hidden=ck["hidden"])
    hd.load_state_dict({k: v.float() for k, v in ck["head"].items()}); hd = hd.to(dev).eval()
    del ck

    files = sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))
    files = files[:a.limit] if a.limit else files
    n = len(files)
    A = np.zeros((n, 4, len(GRID)), np.int32)
    t0 = time.time()
    for i0 in range(0, n, 4):
        ch = files[i0:i0 + 4]
        x = np.stack([(cv2.imread(os.path.join(DATA_DIR, "test_images", f))[:, :, ::-1].astype(np.float32) / 255.0)
                      .transpose(2, 0, 1) for f in ch])
        x = torch.from_numpy(x).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            acc = 0.0
            for v in TTA:
                xv = x if v == "none" else torch.flip(x, [3] if v == "h" else [2])
                p = hd(ad(xv)).float().softmax(1)
                acc = acc + (p if v == "none" else torch.flip(p, [3] if v == "h" else [2]))
            p = (acc / len(TTA)).cpu().numpy()
        for k in range(len(ch)):
            for j in range(4):
                A[i0 + k, j] = areas_ge(cv2.resize(p[k, j + 1], (1600, 256), interpolation=cv2.INTER_LINEAR))
        if (i0 // 4) % 250 == 0:
            done = min(i0 + 4, n); el = time.time() - t0
            print(f"  {done}/{n} | {el / done:.3f} s/img | ETA {(n - done) * el / done / 60:5.1f} min", flush=True)
    sec = time.time() - t0
    np.savez_compressed(os.path.join(OUT, "test_areas_cradio.npz"), grid=GRID, files=np.array(files), solo_cradio=A)

    ref = np.load(REF_NPZ)
    idx = lambda p: int(round(p * 100))
    rate = lambda M, j, k, mina: float((M[:, j, k] >= mina[j]).mean())
    R, S = ref["solo_pseudo01"][:n], ref["solo_soft01"][:n]
    tgt = [rate(R, j, idx(SOLO_PIX[j]), SOLO_MINA) for j in range(4)]
    soft01 = [rate(S, j, idx(SOFT01_PIX[j]), SOLO_MINA) for j in range(4)]
    same = [rate(A, j, idx(SOLO_PIX[j]), SOLO_MINA) for j in range(4)]
    best = [min(range(len(GRID)), key=lambda k: abs(rate(A, j, k, SOLO_MINA) - tgt[j])) for j in range(4)]
    res = {"bundle": os.path.basename(a.bundle), "n_test": n, "seconds": round(sec, 1), "s_per_img": round(sec / n, 4),
           "reference": "solo_pseudo01 (E26's target; soft ViT-B solo matched to it)",
           "reference_pix": SOLO_PIX, "min_area": SOLO_MINA,
           "reference_rate_pct": [round(100 * v, 3) for v in tgt],
           "soft_vitb_rate_at_its_pix_pct": [round(100 * v, 3) for v in soft01],
           "cradio_rate_at_reference_pix_pct": [round(100 * v, 3) for v in same],
           "matched_pix": [float(GRID[b]) for b in best],
           "matched_rate_pct": [round(100 * rate(A, j, best[j], SOLO_MINA), 3) for j in range(4)]}
    json.dump(res, open(os.path.join(OUT, "rate_match.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
