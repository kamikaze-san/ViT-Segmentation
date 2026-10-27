"""Rate-match the calibration arms (E25) to the E23 control before submitting.

Dropping the class weights and the any-pixel targets changes the probability scale,
so the single-model PIX tuned for the old (inflated) models cannot be reused. As in
simplex_rate_match.py: Kaggle-cell inference (bundles, 3-view TTA, fp16, bilinear)
on all 5,506 test images, min_area fixed at the single-model set, pixel threshold
swept per class, and for each arm the threshold whose predicted-positive RATE
equals the control's. The control's rates come from the E23 run, so only the two
new models are inferred. Label-free.
"""
import json, os, time
import numpy as np, cv2

import pseudo_teacher as PT
from pseudo_diag import load
from train_lora import DATA_DIR

OUT = os.path.join(PT.OUTPUTS, "calib_ab")
MODELS = {"soft":   "kaggle_bundle_calib/model_vits_soft10.pt",
          "metric": "kaggle_bundle_calib/model_vits_metric10.pt"}
MINA = PT.E17_MINA
PIXG = np.round(np.arange(0.04, 0.961, 0.02), 3).astype(np.float32)


def main():
    device = "cuda"
    os.makedirs(OUT, exist_ok=True)
    ctl = json.load(open(os.path.join(PT.OUTPUTS, "simplex_ab", "rate_match.json")))
    target = np.array(ctl["control_rate_pct"]) / 100
    files = sorted(os.listdir(os.path.join(DATA_DIR, "test_images"))); n = len(files)
    area = {m: np.zeros((n, 4, len(PIXG)), np.int32) for m in MODELS}
    nets = {m: PT.load_member(rel, device) for m, rel in MODELS.items()}
    t0 = time.time()
    for i0 in range(0, n, 4):
        x = load(os.path.join(DATA_DIR, "test_images"), files[i0:i0 + 4]).to(device)
        for m in MODELS:
            p = PT.member_probs(nets[m], x, PT.TTA, half=True)
            for k in range(len(x)):
                for j in range(4):
                    full = cv2.resize(p[k, j + 1], (1600, 256), interpolation=cv2.INTER_LINEAR)
                    area[m][i0 + k, j] = (full[None] > PIXG[:, None, None]).sum((1, 2))
        if (i0 // 4) % 250 == 0:
            done = i0 + 4; el = time.time() - t0
            print(f"  {min(done, n)}/{n} | ETA {(n - done) * el / done / 60:5.1f} min", flush=True)
    np.savez_compressed(os.path.join(OUT, "test_areas.npz"), pix_grid=PIXG, **{f"area_{m}": v for m, v in area.items()})

    res = {"min_area": MINA, "control_pix": ctl["control_pix"], "control_rate_pct": ctl["control_rate_pct"]}
    ctl_idx = [int(np.argmin(np.abs(PIXG - p))) for p in ctl["control_pix"]]
    print(f"\n{n} test images, 3-view TTA, min_area {MINA} | control rate % {ctl['control_rate_pct']} at PIX {ctl['control_pix']}")
    for m in MODELS:
        rates = (area[m] >= np.array(MINA)[None, :, None]).mean(0)          # (4, len(PIXG))
        same = [round(100 * float(rates[j, ctl_idx[j]]), 3) for j in range(4)]
        best = [int(np.argmin(np.abs(rates[j] - target[j]))) for j in range(4)]
        res[m] = {"rate_at_control_pix_pct": same, "matched_pix": [float(PIXG[b]) for b in best],
                  "matched_rate_pct": [round(100 * float(rates[j, best[j]]), 3) for j in range(4)],
                  "edge_of_grid": [bool(b in (0, len(PIXG) - 1)) for b in best]}
        print(f"  {m:<7} rate at control PIX {same} -> matched PIX {res[m]['matched_pix']} "
              f"(rate {res[m]['matched_rate_pct']}){'  WARNING: grid edge' if any(res[m]['edge_of_grid']) else ''}")
    json.dump(res, open(os.path.join(OUT, "rate_match.json"), "w"), indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
