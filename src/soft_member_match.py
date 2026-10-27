"""Thresholds for the E26 soft-target ViT-B member, solo and swapped into V79.

Soft targets and no class weights change the probability scale (E25: the ViT-S arm
needed PIX ~0.15-0.30 lower to fire as often), so neither the single-model nor the
ensemble-tuned thresholds can be reused. One Kaggle-cell inference pass (bundles,
3-view TTA, fp16, bilinear) over all 5,506 test images gives, label-free:

  solo  new member's per-class PIX matching vitb_lora_warm_pseudo01's firing rate at
        the single-model rule (PIX [.60,.60,.60,.45], min_area [1050,2450,1750,5250])
  swap  V79 with pseudo01 replaced by the new member: per-class PIX matching V79's
        rate at its tuned rule (PIX [.65,.60,.65,.58], min_area [630,1470,1050,3150])

Areas come from a per-map histogram, so the threshold grid is fine (0.01) at no
extra cost and includes 0.45 and 0.58 exactly.
"""
import json, os, time
import numpy as np, cv2

import pseudo_teacher as PT
from pseudo_diag import load
from train_lora import DATA_DIR

OUT = os.path.join(PT.OUTPUTS, "soft_member")
MODELS = {"vitb":     "kaggle_bundle_vitb/model_vitb.pt",
          "vits_r16": "kaggle_bundle_lora/model_vits_lora_r16.pt",
          "lora_r8":  "kaggle_bundle_lora/model_lora_r8.pt",
          "bce16":    "kaggle_bundle_bce/model_bce16.pt",
          "pseudo01": "kaggle_bundle_pseudo/model_vitb_lora_warm_pseudo01.pt",
          "soft01":   "kaggle_bundle_calib/model_vitb_lora_warm_soft_pseudo01.pt"}
BASE = ["vitb", "vits_r16", "lora_r8", "bce16"]
SETS = {"solo_pseudo01": ["pseudo01"], "solo_soft01": ["soft01"],
        "V79": BASE + ["pseudo01"], "V79_swap": BASE + ["soft01"]}
SOLO_PIX, SOLO_MINA = [0.60, 0.60, 0.60, 0.45], [1050, 2450, 1750, 5250]
ENS_PIX, ENS_MINA = [0.65, 0.60, 0.65, 0.58], [630, 1470, 1050, 3150]
EDGES = np.round(np.arange(0.0, 1.0001, 0.01), 2)          # 101 edges, 100 bins
GRID = EDGES[:-1]                                          # area(>= GRID[k])


def areas_ge(p_full):
    counts, _ = np.histogram(p_full, bins=EDGES)
    return np.cumsum(counts[::-1])[::-1]                   # pixels with prob >= GRID[k]


def main():
    device = "cuda"
    os.makedirs(OUT, exist_ok=True)
    files = sorted(os.listdir(os.path.join(DATA_DIR, "test_images"))); n = len(files)
    nets = {m: PT.load_member(rel, device) for m, rel in MODELS.items()}
    A = {s: np.zeros((n, 4, len(GRID)), np.int32) for s in SETS}
    t0 = time.time()
    for i0 in range(0, n, 4):
        x = load(os.path.join(DATA_DIR, "test_images"), files[i0:i0 + 4]).to(device)
        probs = {m: PT.member_probs(net, x, PT.TTA, half=True) for m, net in nets.items()}
        for s, mem in SETS.items():
            avg = sum(probs[m] for m in mem) / len(mem)
            for k in range(len(x)):
                for j in range(4):
                    A[s][i0 + k, j] = areas_ge(cv2.resize(avg[k, j + 1], (1600, 256), interpolation=cv2.INTER_LINEAR))
        if (i0 // 4) % 250 == 0:
            done = i0 + 4; el = time.time() - t0
            print(f"  {min(done, n)}/{n} | ETA {(n - done) * el / done / 60:5.1f} min", flush=True)
    np.savez_compressed(os.path.join(OUT, "test_areas.npz"), grid=GRID, **A)

    idx = lambda p: int(round(p * 100))
    rate = lambda s, j, k, mina: float((A[s][:, j, k] >= mina[j]).mean())
    res = {}
    for name, ref, new, pix, mina in (("solo", "solo_pseudo01", "solo_soft01", SOLO_PIX, SOLO_MINA),
                                      ("swap", "V79", "V79_swap", ENS_PIX, ENS_MINA)):
        tgt = [rate(ref, j, idx(pix[j]), mina) for j in range(4)]
        same = [rate(new, j, idx(pix[j]), mina) for j in range(4)]
        best = [min(range(len(GRID)), key=lambda k: abs(rate(new, j, k, mina) - tgt[j])) for j in range(4)]
        res[name] = {"reference": ref, "reference_pix": pix, "min_area": mina,
                     "reference_rate_pct": [round(100 * v, 3) for v in tgt],
                     "new_rate_at_reference_pix_pct": [round(100 * v, 3) for v in same],
                     "matched_pix": [float(GRID[b]) for b in best],
                     "matched_rate_pct": [round(100 * rate(new, j, best[j], mina), 3) for j in range(4)]}
        r = res[name]
        print(f"\n{name}: {ref} rate % {r['reference_rate_pct']} at PIX {pix} min_area {mina}")
        print(f"  {new} at the same PIX: {r['new_rate_at_reference_pix_pct']}")
        print(f"  {new} matched PIX {r['matched_pix']} -> rate % {r['matched_rate_pct']}")
    json.dump(res, open(os.path.join(OUT, "rate_match.json"), "w"), indent=1)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
