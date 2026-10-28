"""One-minute check of the C-RADIO position fix (E31): do the tap features the adapter returns in train mode equal the
eval-mode ones once RadioBackbone.train_mode_fix() is applied, and how far apart are they without it?

Uses 4 real 256x1600 train strips, the stock backbone (no LoRA), fp32, no grad. Without the fix the train-mode CPE
patch generator re-samples the position grid per sample (and drops it for 10%), so features differ; with it they
should match eval exactly (the backbone has no other train-only behaviour).
"""
import os, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F

import train_full_ft as T
from backbones import build_backbone

# python check_cradio_train_mode.py [backbone folder]   (default $CRADIO_PATH, else ../external/C-RADIOv4-SO400M)
PATH = (sys.argv[1] if len(sys.argv) > 1 else
        os.environ.get("CRADIO_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "external", "C-RADIOv4-SO400M")))
DEV = ("cuda" if torch.cuda.is_available() else
       "mps" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else "cpu")


def per_tap_cos(a, b, dim):
    # (B, 4*dim, gh, gw) -> per-tap mean cosine over tokens
    return [F.cosine_similarity(a[:, i * dim:(i + 1) * dim], b[:, i * dim:(i + 1) * dim], dim=1).mean().item()
            for i in range(a.shape[1] // dim)]


def main():
    torch.manual_seed(0)
    ad = build_backbone("cradio", PATH, DEV)
    norm = {"mean": np.array(ad.norm["mean"], np.float32), "std": np.array(ad.norm["std"], np.float32)}
    ids = pd.read_csv(f"{T.SPLIT_DIR}/train12k_ids.csv")["ImageId"].tolist()[:4]
    x, _, _ = T.read_batch(ids, np.arange(len(ids)), 4, norm)
    x = torch.from_numpy(x).to(DEV)
    print(f"device {DEV} | backbone {PATH}")
    with torch.no_grad():
        ad.module.eval()
        ref = ad(x)
        ad.module.train()
        nofix = [ad(x) for _ in range(2)]
        ad.train_mode_fix()
        fix = ad(x)
    print(f"patch generator training flag after fix: {ad.pos_gen.training} | taps {ad.taps}")
    for k, f in enumerate(nofix):
        print(f"train mode, no fix (draw {k}): max |diff| vs eval {(f - ref).abs().max().item():.4f} | "
              f"per-tap cosine {[round(c, 3) for c in per_tap_cos(f, ref, ad.dim)]}")
    print(f"train mode, with fix:          max |diff| vs eval {(fix - ref).abs().max().item():.2e} | "
          f"per-tap cosine {[round(c, 4) for c in per_tap_cos(fix, ref, ad.dim)]}")
    print("PASS" if torch.equal(fix, ref) else "FAIL: train-mode features still differ from eval")


if __name__ == "__main__":
    main()
