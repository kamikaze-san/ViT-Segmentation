"""Which models actually make DIFFERENT decisions on the target distribution?

The ensemble was built greedily - add whichever candidate scored best next - and
never by asking whether members disagree. E8 tried this on the holdout and the
answer was uninterpretable, because the gains it correlated against were inside
the noise floor. This measures disagreement on the 5,506 COMPETITION TEST images
instead, which needs no labels and is the distribution that actually counts.

Reports, per (image, class) empty/non-empty decision at the submitted thresholds:
  - each model's predicted-positive rate
  - pairwise disagreement between every pair
  - each candidate's disagreement with the current 5-member ensemble's decisions

A useful new member has to be BOTH good (known from its own submission) and
different (measured here). Greedy addition only ever checked the first.
"""
import argparse, itertools, os
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoImageProcessor

from spatial_head import build, N_CLASSES
from train_lora import MODEL_ID, DATA_DIR, CKPT_DIR, GRID

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\complement"
CLASSES = (1, 2, 3, 4)
PIX = [0.60, 0.60, 0.60, 0.45]          # the 0.90711 operating point
MINA = [1050, 2450, 1750, 5250]

MODELS = {   # name: (checkpoint, is_ensemble_member)
    "vitb":            ("fullft_lr0.0001_g0.85_h1024_vitb_blackout_ep16.pt", True),
    "vits_lora_r16":   ("fullft_lr0.0001_g0.85_h1024_vits_lora_r16_ep10.pt", True),
    "lora_r8":         ("merged_lora_r8.pt", True),
    "bce16":           ("fullft_lr0.0001_g0.85_h1024_bce_blackout_ep16.pt", True),
    "vitb_lora_warm":  ("fullft_lr0.0001_g0.85_h1024_vitb_lora_warm_ep10.pt", True),
    "blackout":        ("fullft_lr0.0001_g0.85_h1024_blackout_ep16.pt", False),
    "12k_ep16":        ("fullft_lr0.0001_g0.85_h1024_12k_ep16.pt", False),
    "vits_ft10":       ("fullft_lr0.0001_g0.85_h1024_vits_ft10_ep10.pt", False),
    "vits_taps":       ("fullft_lr0.0001_g0.85_h1024_vits_taps0371_ep10.pt", False),
    "vitl_lora":       ("fullft_lr0.0001_g0.85_h1024_vitl_lora_ep10.pt", False),
}


@torch.no_grad()
def decisions(name, ckpt, files, device, bs=4):
    """(n_images, 4) bool: does this model predict class c present? Cached."""
    cache = f"{OUT}/dec_{name}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    ck = torch.load(f"{CKPT_DIR}/{ckpt}", map_location=device, weights_only=False)
    mid = ck.get("model_id") or MODEL_ID
    m = AutoModel.from_pretrained(mid).to(device).eval(); m.load_state_dict(ck["backbone"])
    taps = tuple(ck["tap_layers"]); dim = m.config.hidden_size
    h = build(ck.get("head_arch", "conv"), dim=dim * len(taps), upscale=4,
              hidden=ck["hidden"]).to(device).eval()
    h.load_state_dict(ck["head"])
    sig = ck.get("loss") == "bce_dice"
    proc = AutoImageProcessor.from_pretrained(mid)
    mean = np.array(proc.image_mean, np.float32); std = np.array(proc.image_std, np.float32)
    ns = 1 + getattr(AutoConfig.from_pretrained(mid), "num_register_tokens", 0)

    out = np.zeros((len(files), len(CLASSES)), bool)
    for i0 in range(0, len(files), bs):
        ch = files[i0:i0 + bs]
        x = np.stack([(((cv2.imread(f"{DATA_DIR}/test_images/{f}")[:, :, ::-1]
                         .astype(np.float32) / 255.0) - mean) / std).transpose(2, 0, 1) for f in ch])
        o = m(pixel_values=torch.from_numpy(x).to(device), output_hidden_states=True)
        tok = torch.cat([F.layer_norm(o.hidden_states[t + 1][:, ns:, :],
                                      (o.hidden_states[t + 1].shape[-1],)) for t in taps], -1)
        lg = h(tok.permute(0, 2, 1).reshape(len(ch), -1, GRID[0], GRID[1])).float()
        p = torch.sigmoid(lg) if sig else lg.softmax(1)
        p = p.cpu().numpy()
        for k in range(len(ch)):
            for j, c in enumerate(CLASSES):
                full = cv2.resize(p[k, c], (1600, 256), interpolation=cv2.INTER_LINEAR) > PIX[j]
                out[i0 + k, j] = full.sum() >= MINA[j]
        if (i0 + bs) % 1600 < bs:
            print(f"    {name}: {min(i0+bs,len(files))}/{len(files)}", flush=True)
    np.save(cache, out); del m, h; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(os.listdir(f"{DATA_DIR}/test_images"))[:a.limit]
    print(f"{len(files)} competition test images | thresholds pix {PIX} min_area {MINA}\n")

    dec = {}
    for n, (c, _) in MODELS.items():
        print(f"### {n}")
        dec[n] = decisions(n, c, files, "cuda")

    print(f"\n{'model':<16}{'rate':>8}{'member':>9}")
    for n in MODELS:
        print(f"{n:<16}{100*dec[n].mean():>7.2f}%{'yes' if MODELS[n][1] else '':>9}")

    names = list(MODELS)
    print(f"\nPAIRWISE DISAGREEMENT (% of the {len(files)*4} image-class decisions)")
    print(f"{'':<16}" + "".join(f"{n[:7]:>9}" for n in names))
    for a_ in names:
        row = "".join(f"{100*(dec[a_] != dec[b]).mean():>9.2f}" for b in names)
        print(f"{a_:<16}{row}")

    mem = [n for n in names if MODELS[n][1]]
    print(f"\nDISAGREEMENT WITH THE 5 CURRENT MEMBERS (mean over members)")
    print(f"{'candidate':<16}{'vs members':>12}{'in ensemble':>13}")
    for n in names:
        others = [m for m in mem if m != n]
        v = np.mean([(dec[n] != dec[m]).mean() for m in others])
        print(f"{n:<16}{100*v:>11.2f}%{'yes' if MODELS[n][1] else 'no':>13}")
    print("\nA useful new member must be BOTH good (its own submitted score) and")
    print("different (this column). Greedy addition only ever checked the first.")


if __name__ == "__main__":
    main()
