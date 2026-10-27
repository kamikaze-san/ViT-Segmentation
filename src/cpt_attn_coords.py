"""H-A vs H-B (E27): is the strip-only broadening of CPT attention caused by the RoPE coordinate range or by token count?

Same measurement as cpt_attn_diag.py (rel_dist = attention distance / uniform distance), but the RoPE patch
coordinates are rescaled per axis so token count and coordinate spacing are decoupled:
  A  14x14 crops, x-coordinates scaled by 14/100: 196 tokens, horizontal gap 0.020 (the strip's)
  B  16x100 strips, x-coordinates scaled by 100/14: 1,600 tokens, horizontal gap 0.143 (a crop's)
  controls (from cpt_attn_diag.py): crop with its own coordinates, strip with its own coordinates
Reading: CPT-minus-stock rel_dist in layers 3-4.
  A broadens (like the strip) -> coordinate range (H-A).   B stays narrow (like the crop) -> coordinate range.
  A stays like the crop, B broadens -> token count (H-B).
"""
import os, json
import numpy as np, pandas as pd, cv2, torch
from transformers import AutoModel
import transformers.models.dinov3_vit.modeling_dinov3_vit as M

from train_lora import DATA_DIR, SPLIT_DIR, CKPT_DIR
from cpt_attn_diag import MID, N, MEAN, STD, run

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\cpt\attn_coords.json"
_orig = M.get_patches_center_coordinates
SCALE = [1.0, 1.0]                                                    # (y, x)


def patched(h, w, *a, **k):
    c = _orig(h, w, *a, **k)
    return c * torch.tensor(SCALE, dtype=c.dtype, device=c.device)


M.get_patches_center_coordinates = patched


def main():
    stock = AutoModel.from_pretrained(MID, attn_implementation="eager").cuda().eval()
    cpt = AutoModel.from_pretrained(MID, attn_implementation="eager")
    cpt.load_state_dict(torch.load(os.path.join(CKPT_DIR, "cpt_vitb_ep5.pt"), map_location="cpu", weights_only=False)["backbone"])
    cpt = cpt.cuda().eval()
    ns = 1 + getattr(stock.config, "num_register_tokens", 0)
    ids = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()[:N]
    load = lambda f: ((cv2.imread(os.path.join(DATA_DIR, "train_images", f))[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    strips = [torch.from_numpy(load(f))[None] for f in ids]
    crops = [torch.from_numpy(np.stack([load(f)[:, 16:240, x0:x0 + 224] for x0 in range(0, 1600 - 224 + 1, 224)])) for f in ids]

    cases = {"A_crop_with_strip_x_coords": (crops, 14, 14, [1.0, 14 / 100]),
             "B_strip_with_crop_x_coords": (strips, 16, 100, [1.0, 100 / 14]),
             "ctrl_crop": (crops, 14, 14, [1.0, 1.0]),
             "ctrl_strip": (strips, 16, 100, [1.0, 1.0])}
    res = {}
    for name, (bat, gh, gw, sc) in cases.items():
        SCALE[:] = sc
        for m, mdl in (("stock", stock), ("cpt", cpt)):
            res[f"{name}/{m}"] = run(mdl, bat, gh, gw, ns)[:, 0].tolist()      # rel_dist per layer
            print("done", name, m, flush=True)
    SCALE[:] = [1.0, 1.0]
    json.dump({"n_images": N, "metric": "rel_dist per layer", **res}, open(OUT, "w"), indent=1)

    print("\nrel_dist stock -> CPT (diff), layers 1..6 and 11")
    for name in cases:
        a, b = np.array(res[f"{name}/stock"]), np.array(res[f"{name}/cpt"])
        print(f"{name:<28}" + " ".join(f"L{l}: {a[l]:.3f}>{b[l]:.3f}({b[l] - a[l]:+.3f})" for l in (1, 2, 3, 4, 5, 6, 11)))
    print("saved", OUT)


if __name__ == "__main__":
    main()
