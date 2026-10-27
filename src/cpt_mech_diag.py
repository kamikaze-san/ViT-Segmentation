"""E27 mechanism check: why did continued pretraining make frozen features slightly worse?

Stock DINOv3 ViT-B vs cpt_vitb_ep5.pt, no training, holdout strips at native 256x1600.
  1 patch-CLS cosine        DINOv3 paper (Sec. 3.2, 4.1): rises as dense quality degrades
  2 locality                neighbour vs distant patch cosine, and mean within-image patch cosine
  3 between-image share     of patch-feature variance (image-specific component in every patch)
  4 defect AUC              within an image, do defect patches stand out from that image's clean steel
  5 drift full vs crop      CPT-vs-stock cosine on full strips vs 224x224 crops (resolution hypothesis)
Metrics 2-5 use per-patch LayerNorm'd features, which is what the segmentation head reads.
"""
import os, json, copy
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
from transformers import AutoModel

from spatial_head import soft_targets_at_stride
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR, CKPT_DIR

MID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
CPT = "cpt_vitb_ep5.pt"
LAYERS = ("2", "5", "8", "11", "out")            # head taps + final normed output
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
N, BS, CROP, CY = 500, 4, 224, 16
OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\cpt\mech_diag.json"


def load(paths, idx):
    return torch.from_numpy(np.stack([((cv2.imread(paths[i])[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD)
                                      .transpose(2, 0, 1) for i in idx]))


@torch.no_grad()
def feats(model, x):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(pixel_values=x, output_hidden_states=True)
    r = {str(t): o.hidden_states[t + 1].float() for t in (2, 5, 8, 11)}
    r["out"] = o.last_hidden_state.float()
    return r


def ln_unit(p):
    z = F.layer_norm(p, (p.shape[-1],))
    return z, F.normalize(z, dim=-1)


class Acc:
    def __init__(self):
        self.lists = {}
        self.var = {L: [None, 0.0, 0.0, 0] for L in LAYERS}       # sum_z, sum_sq, sum_mean_sq, n_tok
        self.auc = {L: {c: ([], []) for c in range(1, 5)} for L in LAYERS}

    def add(self, key, v):
        self.lists.setdefault(key, []).append(float(v))

    def mean(self, key):
        return float(np.mean(self.lists[key]))


def per_image(acc, f, ns, gh, gw, tag, targets=None):
    """f: dict layer -> (B, ns+P, D). targets: list of (5,P) soft targets or None."""
    for L in LAYERS:
        h = f[L]
        for b in range(h.shape[0]):
            cls, p = h[b, 0], h[b, ns:]
            acc.add(f"{tag}/cls_cos/{L}", F.cosine_similarity(p, cls[None], dim=-1).mean())
            z, zn = ln_unit(p)
            P = zn.shape[0]
            acc.add(f"{tag}/pair_mean/{L}", ((zn.sum(0).square().sum() - P) / (P * (P - 1))))
            g = zn.view(gh, gw, -1)
            acc.add(f"{tag}/near/{L}", 0.5 * ((g[:, :-1] * g[:, 1:]).sum(-1).mean() + (g[:-1] * g[1:]).sum(-1).mean()))
            if gw > 50:
                acc.add(f"{tag}/far10/{L}", (g[:, :-10] * g[:, 10:]).sum(-1).mean())
                acc.add(f"{tag}/far50/{L}", (g[:, :-50] * g[:, 50:]).sum(-1).mean())
            if targets is None:
                continue
            v = acc.var[L]
            v[0] = z.sum(0) if v[0] is None else v[0] + z.sum(0)
            v[1] += float(z.square().sum()); v[2] += P * float(z.mean(0).square().sum()); v[3] += P
            t = targets[b]
            clean = t[0] >= 0.999
            if clean.sum() < 20:
                continue
            c = F.normalize(zn[torch.from_numpy(clean).cuda()].mean(0), dim=0)
            s = (1 - zn @ c).cpu().numpy()
            for k in range(1, 5):
                d = t[k] >= 0.5
                if d.sum():
                    acc.auc[L][k][0].append(s[d]); acc.auc[L][k][1].append(s[clean])


def auc(pos, neg):
    if not pos:
        return None
    pos, neg = np.concatenate(pos), np.concatenate(neg)
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    dev = "cuda"
    stock = AutoModel.from_pretrained(MID).to(dev).eval()
    cpt = copy.deepcopy(stock)
    cpt.load_state_dict(torch.load(os.path.join(CKPT_DIR, CPT), map_location="cpu", weights_only=False)["backbone"])
    cpt.eval()
    ns = 1 + getattr(stock.config, "num_register_tokens", 0)
    ids = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()[:N]
    paths = [os.path.join(DATA_DIR, "train_images", f) for f in ids]
    acc = {"stock": Acc(), "cpt": Acc()}
    drift = {f"{m}/{L}": [] for m in ("full", "crop") for L in LAYERS}

    for i0 in range(0, len(ids), BS):
        idx = list(range(i0, min(i0 + BS, len(ids))))
        x = load(paths, idx).to(dev)
        tg = [soft_targets_at_stride(cv2.imread(os.path.join(MASK_DIR, os.path.splitext(ids[i])[0] + ".png"),
                                                cv2.IMREAD_UNCHANGED), 16).reshape(5, -1) for i in idx]
        xc = torch.cat([x[:, :, CY:CY + CROP, x0:x0 + CROP] for x0 in range(0, 1600 - CROP + 1, CROP)])
        for mode, inp, gh, gw, t in (("full", x, 16, 100, tg), ("crop", xc, 14, 14, None)):
            fs, fc = feats(stock, inp), feats(cpt, inp)
            per_image(acc["stock"], fs, ns, gh, gw, mode, t)
            per_image(acc["cpt"], fc, ns, gh, gw, mode, t)
            for L in LAYERS:
                a, b = ln_unit(fs[L][:, ns:])[0], ln_unit(fc[L][:, ns:])[0]
                drift[f"{mode}/{L}"].append(float(F.cosine_similarity(a, b, dim=-1).mean()))
        if i0 % 100 == 0:
            print(f"{i0 + len(idx)}/{len(ids)}", flush=True)

    res = {"n_images": len(ids), "layers": LAYERS}
    for m in ("stock", "cpt"):
        A, r = acc[m], {}
        for key in A.lists:
            r[key] = round(A.mean(key), 4)
        for L in LAYERS:
            sz, ssq, smq, n = A.var[L]
            mu2 = float((sz / n).square().sum())
            r[f"between_share/{L}"] = round((smq / n - mu2) / (ssq / n - mu2), 4)
            for k in range(1, 5):
                v = auc(*A.auc[L][k])
                r[f"auc_c{k}/{L}"] = None if v is None else round(v, 4)
        res[m] = r
    res["drift"] = {k: round(float(np.mean(v)), 4) for k, v in drift.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    def row(name, key, fmt="{:.3f}"):
        cells = []
        for L in LAYERS:
            a, b = res["stock"].get(key.format(L)), res["cpt"].get(key.format(L))
            cells.append("   n/a      " if a is None else f"{fmt.format(a)}/{fmt.format(b)} ({b - a:+.3f})")
        print(f"{name:<26}" + " | ".join(cells))
    print("\nstock/CPT (diff) per layer: " + " | ".join(LAYERS))
    row("patch-CLS cos (full)", "full/cls_cos/{}")
    row("patch-CLS cos (crop)", "crop/cls_cos/{}")
    row("mean patch-pair cos", "full/pair_mean/{}")
    row("neighbour cos", "full/near/{}")
    row("cos 10 patches apart", "full/far10/{}")
    row("cos 50 patches apart", "full/far50/{}")
    row("between-image share", "between_share/{}")
    for k in range(1, 5):
        row(f"defect AUC c{k}", f"auc_c{k}/{{}}")
    print("drift cos CPT vs stock, full / crop: " + " | ".join(
        f"{res['drift']['full/' + L]:.3f}/{res['drift']['crop/' + L]:.3f}" for L in LAYERS))
    print("saved", OUT)


if __name__ == "__main__":
    main()
