"""Put the simplex arm and the control arm on the same footing before submitting.

The simplex head scores pixels by cosine against fixed class vertices, scaled by a
learned temperature, so its probability scale is NOT the control's. Submitting both
at the single-model PIX would confound the head change with a threshold mismatch.

This measures, under the Kaggle cell's exact inference (bundles, 3-view TTA, fp16,
bilinear), the per-class predicted-positive RATE of each arm over all 5,506 test
images across a grid of pixel thresholds, with min_area fixed at the single-model
set. The control keeps its standard PIX; for the simplex arm it reports the pixel
threshold whose rate matches the control's, per class. Label-free: no test labels
are used, only how often each model fires.
"""
import json, os, time
import numpy as np, pandas as pd, cv2

import pseudo_teacher as PT
from pseudo_diag import load
from train_lora import DATA_DIR

OUT = os.path.join(PT.OUTPUTS, "simplex_ab")
MODELS = {"ctl":  "kaggle_bundle_simplex/model_vits_ctl10.pt",
          "splx": "kaggle_bundle_simplex/model_vits_splx10.pt"}
CTL_PIX, MINA = PT.E17_PIX, PT.E17_MINA          # the single-model rule
PIXG = np.round(np.arange(0.20, 0.961, 0.02), 3).astype(np.float32)


def main():
    device = "cuda"
    os.makedirs(OUT, exist_ok=True)
    files = sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))
    n = len(files)
    area = {m: np.zeros((n, 4, len(PIXG)), np.int32) for m in MODELS}
    t0 = time.time()
    nets = {m: PT.load_member(rel, device) for m, rel in MODELS.items()}
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

    rate = lambda m, j, pi: float((area[m][:, j, pi] >= MINA[j]).mean())
    ctl_idx = [int(np.argmin(np.abs(PIXG - p))) for p in CTL_PIX]
    res = {"min_area": MINA, "control_pix": CTL_PIX, "control_rate_pct": [], "simplex_matched_pix": [],
           "simplex_rate_at_control_pix_pct": [], "simplex_matched_rate_pct": []}
    print(f"\n{n} test images, 3-view TTA, min_area {MINA}")
    print(f"{'cls':>4}{'control pix':>13}{'control rate':>14}{'simplex @same pix':>19}{'simplex matched pix':>21}{'matched rate':>14}")
    for j in range(4):
        cr = rate("ctl", j, ctl_idx[j])
        sr_same = rate("splx", j, ctl_idx[j])
        rates = np.array([rate("splx", j, pi) for pi in range(len(PIXG))])
        best = int(np.argmin(np.abs(rates - cr)))
        res["control_rate_pct"].append(round(100 * cr, 3))
        res["simplex_rate_at_control_pix_pct"].append(round(100 * sr_same, 3))
        res["simplex_matched_pix"].append(float(PIXG[best]))
        res["simplex_matched_rate_pct"].append(round(100 * rates[best], 3))
        print(f"{j + 1:>4}{CTL_PIX[j]:>13.2f}{100 * cr:>13.2f}%{100 * sr_same:>18.2f}%"
              f"{PIXG[best]:>21.2f}{100 * rates[best]:>13.2f}%")
    json.dump(res, open(os.path.join(OUT, "rate_match.json"), "w"), indent=1)
    print(f"\nsubmit the simplex arm with PIX {res['simplex_matched_pix']} and MINA {MINA}")
    print(f"(control uses PIX {CTL_PIX} MINA {MINA}) | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
