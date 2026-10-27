"""Label-free c3 rate match: do pseudo-heavy ensembles lose private score because
they fire c3 more?

E21 found that each pseudo swap both lowered member diversity AND raised the
ensemble's c3 firing on the images the teacher could not label (V79 20.58% ->
V83 22.43%, single-view). This separates the two. Raise only V83's (and V82's) c3
min_area until its c3 positive rate on the full test set equals V79's, under the
Kaggle cell's exact inference (bundles, 3-view TTA, fp16, bilinear), and submit:

  recovers to >= V79  -> the loss was c3 over-firing, fixable by a c3 threshold
  stays below V79     -> the loss is diversity; more swaps are a dead end

Matching the RATE does not guarantee matching the IMAGES, so it also reports how
many c3 decisions still differ from V79 after the match. Trains nothing.
"""
import json, os, time
import numpy as np, pandas as pd, cv2, torch

import pseudo_teacher as PT
from pseudo_diag import MODELS, ENSEMBLES, TUNED_PIX, TUNED_MINA, load
from train_lora import DATA_DIR, SPLIT_DIR

OUT = os.path.join(PT.OUTPUTS, "pseudo_diag")
C3_PIX = [0.60, 0.65, 0.70, 0.75]


def main():
    device = "cuda"
    os.makedirs(OUT, exist_ok=True)
    files = sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))
    n = len(files)
    pseudo = set(pd.read_csv(os.path.join(SPLIT_DIR, "pseudo_ids.csv"))["ImageId"])
    in_ps = np.array([f in pseudo for f in files])
    nets = {m: PT.load_member(rel, device) for m, rel in MODELS.items()}

    area_t = {e: np.zeros((n, 4), np.int32) for e in ENSEMBLES}          # area at the tuned pix
    area3 = {e: np.zeros((n, len(C3_PIX)), np.int32) for e in ENSEMBLES}  # c3 area at several pix
    t0 = time.time()
    for i0 in range(0, n, 4):
        x = load(os.path.join(DATA_DIR, "test_images"), files[i0:i0 + 4]).to(device)
        probs = {m: PT.member_probs(nets[m], x, PT.TTA, half=True) for m in MODELS}
        for e, mem in ENSEMBLES.items():
            avg = sum(probs[m] for m in mem) / len(mem)
            for k in range(len(x)):
                for j in range(4):
                    full = cv2.resize(avg[k, j + 1], (1600, 256), interpolation=cv2.INTER_LINEAR)
                    area_t[e][i0 + k, j] = (full > TUNED_PIX[j]).sum()
                    if j == 2:
                        area3[e][i0 + k] = [(full > p).sum() for p in C3_PIX]
        if (i0 // 4) % 250 == 0:
            done = i0 + 4; el = time.time() - t0
            print(f"  test {min(done, n)}/{n} | ETA {(n - done) * el / done / 60:5.1f} min", flush=True)
    np.savez_compressed(os.path.join(OUT, "c3_rate_match.npz"), in_pseudo=in_ps,
                        **{f"area_t_{e}": v for e, v in area_t.items()}, **{f"area3_{e}": v for e, v in area3.items()})

    dec = {e: area_t[e] >= np.array(TUNED_MINA) for e in ENSEMBLES}
    res = {"rates_at_tuned": {}, "match": {}}
    print(f"\n{n} test images (3-view TTA, fp16, bilinear) - positive rate % at tuned thresholds")
    for e in ENSEMBLES:
        allr = (100 * dec[e].mean(0)).round(2).tolist()
        nps = (100 * dec[e][~in_ps].mean(0)).round(2).tolist()
        res["rates_at_tuned"][e] = {"all": allr, "non_pseudo": nps}
        print(f"  {e}: all {allr} | non-pseudo images {nps}")

    target = dec["V79"][:, 2].mean()
    print(f"\nc3 target = V79's c3 rate {100 * target:.2f}% ({int(dec['V79'][:, 2].sum())} images)")
    for e in ("V82", "V83"):
        a = area_t[e][:, 2]
        mina = next(m for m in range(TUNED_MINA[2], 40001, 10) if (a >= m).mean() <= target)
        new = a >= mina
        v = dec["V79"][:, 2]; old = dec[e][:, 2]
        r = {"c3_min_area": mina, "c3_rate_before": round(100 * float(old.mean()), 2),
             "c3_rate_after": round(100 * float(new.mean()), 2),
             "c3_decisions_differ_from_V79_before": int((old != v).sum()),
             "c3_decisions_differ_from_V79_after": int((new != v).sum()),
             "after_only_this_ens_positive": int((new & ~v).sum()), "after_only_V79_positive": int((v & ~new).sum()),
             "non_pseudo_c3_rate_after": round(100 * float(new[~in_ps].mean()), 2)}
        for pi, p in enumerate(C3_PIX):   # the other knob, for reference: pix alone at the tuned min_area
            r[f"c3_rate_if_pix_{p}"] = round(100 * float((area3[e][:, pi] >= TUNED_MINA[2]).mean()), 2)
        res["match"][e] = r
        print(f"  {e}: c3 min_area {TUNED_MINA[2]} -> {mina} | rate {r['c3_rate_before']}% -> {r['c3_rate_after']}% | "
              f"c3 decisions differing from V79 {r['c3_decisions_differ_from_V79_before']} -> "
              f"{r['c3_decisions_differ_from_V79_after']} (only {e} {r['after_only_this_ens_positive']}, "
              f"only V79 {r['after_only_V79_positive']}) | non-pseudo c3 {r['non_pseudo_c3_rate_after']}%")
        print(f"       for reference, pix-only alternative at min_area {TUNED_MINA[2]}: "
              + ", ".join(f"pix {p} -> {r[f'c3_rate_if_pix_{p}']}%" for p in C3_PIX))
    json.dump(res, open(os.path.join(OUT, "c3_rate_match.json"), "w"), indent=1)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
