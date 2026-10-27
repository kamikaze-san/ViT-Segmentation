"""Label-free complementarity on the real test distribution (where false alarms live).

For every encoder pair, using the per-class presence probes' scores on the 5,506 competition test strips:
  rank correlation on all strips, and on the strips where the V68 teacher ensemble fires (outputs/pseudo/ens_dec.npy:
  the candidate decisions, which is where false alarms come from), and the decision-disagreement rate when each
  encoder is thresholded to fire as often as the ensemble does for that class.
Disagreement between a new encoder and DINOv3-B only matters if it exceeds the DINOv3-B vs DINOv3-L/H control.
"""
import json, os, itertools
import numpy as np
from scipy.stats import spearmanr

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\encoder_probe"
ENS = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\pseudo\ens_dec.npy"


def main():
    names = [n for n in ("dinov3", "dinov3l", "dinov3h", "qwenvit", "qwen38", "cradio_so", "cradio_h")
             if os.path.exists(os.path.join(OUT, f"{n}_test_presence.npy"))]
    S = {n: np.load(os.path.join(OUT, f"{n}_test_presence.npy")) for n in names}
    ens = np.load(ENS).astype(bool)
    ens = ens.reshape(len(ens), -1)[:, -4:] if ens.ndim > 2 or ens.shape[1] != 4 else ens
    rate = ens.mean(0)
    dec = {n: np.stack([S[n][:, c] >= np.quantile(S[n][:, c], 1 - rate[c]) for c in range(4)], 1) for n in names}
    res = {"encoders": names, "ensemble_fire_rate_c1_c4": [round(float(r), 4) for r in rate], "pairs": {}}
    for a, b in itertools.combinations(names, 2):
        r = {}
        for c in range(4):
            fire = ens[:, c]
            r[f"c{c + 1}"] = {"corr_all": round(float(spearmanr(S[a][:, c], S[b][:, c])[0]), 3),
                              "corr_where_ensemble_fires": round(float(spearmanr(S[a][fire, c], S[b][fire, c])[0]), 3) if fire.sum() > 5 else None,
                              "decision_disagreement": round(float((dec[a][:, c] != dec[b][:, c]).mean()), 4)}
        res["pairs"][f"{a}|{b}"] = r
    json.dump(res, open(os.path.join(OUT, "test_complementarity.json"), "w"), indent=1)
    print("fire rates", res["ensemble_fire_rate_c1_c4"])
    print(f"{'pair':28s} " + " | ".join(f"c{c} corr_all/fires/disagree" for c in range(1, 5)))
    for p, r in res["pairs"].items():
        print(f"{p:28s} " + " | ".join(f"{r[f'c{c}']['corr_all']:.2f}/{r[f'c{c}']['corr_where_ensemble_fires']}/{r[f'c{c}']['decision_disagreement']:.3f}" for c in range(1, 5)))


if __name__ == "__main__":
    main()
