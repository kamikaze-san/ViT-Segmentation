"""H-B (E27): does continued pretraining change how far attention reaches, on strips vs crops?

Stock DINOv3 ViT-B vs cpt_vitb_ep5.pt, eager attention so the maps are available, no training.
Per layer, over patch queries -> patch keys (renormalised; special-token mass reported separately):
  rel_dist   mean attention distance / mean distance under uniform attention (grid-size independent locality;
             0 = fully local, 1 = uniform)
  coord_dist mean distance in RoPE coordinate units (box normalised to [-1,1]) - what the model "feels"
  ent        entropy / log(P)
  far10      attention mass on patches more than 10 patches away
  special    mass on CLS + register tokens (before renormalising)
Geometries: full 256x1600 strips (16x100) and 224x224 crops cut from the same strips (14x14).
Prediction of H-B: CPT attends farther than stock on strips but not on crops.
"""
import os, json, copy
import numpy as np, pandas as pd, cv2, torch
from transformers import AutoModel

from train_lora import DATA_DIR, SPLIT_DIR, CKPT_DIR

MID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
N = 100
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\cpt\attn_diag.json"
STATS = ("rel_dist", "coord_dist", "ent", "far10", "special")


def grid_dists(gh, gw):
    r, c = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    r, c = r.ravel().astype(np.float32), c.ravel().astype(np.float32)
    dr, dc = r[:, None] - r[None], c[:, None] - c[None]
    d = np.sqrt(dr ** 2 + dc ** 2)
    dco = np.sqrt((dr * 2 / gh) ** 2 + (dc * 2 / gw) ** 2)
    return torch.from_numpy(d).cuda(), torch.from_numpy(dco).cuda()


@torch.no_grad()
def layer_stats(att, ns, D, Dco):
    """att: (B, H, T, T). -> (n_layer_stats,) averaged over batch, heads, patch queries."""
    A = att.float()
    special = A[:, :, ns:, :ns].sum(-1).mean().item()
    Ap = A[:, :, ns:, ns:]
    Ap = Ap / Ap.sum(-1, keepdim=True).clamp_min(1e-9)
    P = Ap.shape[-1]
    dist = (Ap * D).sum(-1).mean().item()
    rel = dist / D.mean().item()
    coord = (Ap * Dco).sum(-1).mean().item()
    ent = (-(Ap * Ap.clamp_min(1e-12).log()).sum(-1) / np.log(P)).mean().item()
    far10 = (Ap * (D > 10)).sum(-1).mean().item()
    return [rel, coord, ent, far10, special]


def run(model, batches, gh, gw, ns):
    D, Dco = grid_dists(gh, gw)
    acc, n = None, 0
    for x in batches:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            atts = model(pixel_values=x.cuda(), output_attentions=True).attentions
        s = np.array([layer_stats(a, ns, D, Dco) for a in atts])
        acc = s if acc is None else acc + s; n += 1
    return acc / n


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
    res = {}
    for name, bat, gh, gw in (("strip", strips, 16, 100), ("crop", crops, 14, 14)):
        for m, mdl in (("stock", stock), ("cpt", cpt)):
            res[f"{name}/{m}"] = run(mdl, bat, gh, gw, ns).tolist()
            print("done", name, m, flush=True)
    json.dump({"n_images": N, "stats": STATS, **res}, open(OUT, "w"), indent=1)

    print("\nlayer-by-layer, stock -> CPT (diff)")
    for name in ("strip", "crop"):
        a, b = np.array(res[f"{name}/stock"]), np.array(res[f"{name}/cpt"])
        print(f"\n== {name} ==")
        for si, s in enumerate(STATS):
            print(f"{s:<11}" + " ".join(f"L{l:>2} {a[l, si]:.3f}>{b[l, si]:.3f}({b[l, si] - a[l, si]:+.3f})" for l in range(12)))
    print("\nsaved", OUT)


if __name__ == "__main__":
    main()
