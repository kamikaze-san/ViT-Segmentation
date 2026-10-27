"""Frozen-encoder comparison: is a VLM's image encoder as good as DINOv3 for our defects, and does it add information?

Same protocol for every encoder, frozen, 4 layers tapped and LayerNorm'd per patch (as our HeadNet reads them):
  probe     linear patch classifier (5 classes, soft targets at the encoder's patch stride) trained on 800 train12k
            strips (all defect cells + 150 background cells each), scored as per-class Dice on all 500 holdout strips
  presence  per-class logistic regression on pooled [mean || max] features, trained on 3,000 train12k strips,
            AUC on the 500 holdout strips; holdout scores saved for the complementarity check (encoder_compare.py)
Encoders: dinov3 (ViT-B/16, taps 2/5/8/11), dinov3l (ViT-L/16, taps 5/11/17/23, the size-matched
comparison: Qwen's ViT is 24 blocks x 1024 like ViT-L), qwenvit (Qwen3-VL-2B ViT from the WeVisOne-VL-2B weights, taps
5/11/17/23 captured at the merger inputs and un-shuffled to row-major, as WeVisOne's own decoder reads them).
"""
import argparse, json, os, time
import numpy as np, pandas as pd, cv2, torch
import torch.nn as nn, torch.nn.functional as F

from spatial_head import soft_targets_at_stride
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR

ROOT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal"
OUT = os.path.join(ROOT, "outputs", "encoder_probe")
N_PROBE, N_POOL, BG_PER_IMG = 800, 3000, 150


def mask_of(f):
    return cv2.imread(os.path.join(MASK_DIR, os.path.splitext(f)[0] + ".png"), cv2.IMREAD_UNCHANGED)


def taps_for(n):
    """Our 12-block taps (2, 5, 8, 11) scaled to an n-block encoder: the same relative depths for every model."""
    return tuple(int(round((t + 1) * n / 12)) - 1 for t in (2, 5, 8, 11))


class Radio:
    """NVIDIA C-RADIOv4 (distilled from DINOv3-7B, SAM3, SigLIP2-g). Input in [0, 1]; the model normalises itself."""
    stride = 16

    def __init__(self, folder):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(folder, trust_remote_code=True).cuda().eval()
        self.idx = list(taps_for(len(self.m.radio_model.model.blocks)))
        print(f"radio {os.path.basename(folder)}: {len(self.m.radio_model.model.blocks)} blocks, taps {self.idx}", flush=True)

    @torch.no_grad()
    def __call__(self, rgb):
        x = torch.from_numpy(rgb.astype(np.float32).transpose(2, 0, 1)[None] / 255.0).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = self.m.radio_model.forward_intermediates(x, indices=self.idx, output_fmt="NLC", intermediates_only=True)
        gh, gw = rgb.shape[0] // 16, rgb.shape[1] // 16
        assert outs[0].shape[1] == gh * gw, f"radio returned {outs[0].shape[1]} tokens for a {gh}x{gw} grid"
        return torch.cat([F.layer_norm(o[0].float(), (o.shape[-1],)) for o in outs], -1), (gh, gw)


class Qwen38:
    """The Qwen3.8-27B vision encoder (27 x 1152, SigLIP2-SO400M based), loaded alone from the one shard holding it."""
    stride = 16

    def __init__(self):
        from transformers import AutoConfig, AutoImageProcessor, Qwen3_5VisionModel
        from safetensors import safe_open
        path = os.path.join(ROOT, "external", "Qwen3.8-27B-vision")
        cfg = AutoConfig.from_pretrained(path)
        self.v = Qwen3_5VisionModel(cfg.vision_config)
        sd = {}
        with safe_open(os.path.join(path, "model-00001-of-00018.safetensors"), "pt") as f:
            for k in f.keys():
                if "visual." in k:
                    sd[k.split("visual.", 1)[1]] = f.get_tensor(k)
        miss, unexp = self.v.load_state_dict(sd, strict=False)
        print(f"qwen3.8 vision: loaded {len(sd)} tensors | missing {len(miss)} {miss[:3]} | unexpected {len(unexp)} {unexp[:3]}", flush=True)
        assert len(miss) == 0, "vision weights incomplete"
        self.v = self.v.to("cuda", torch.bfloat16).eval()
        self.proc = AutoImageProcessor.from_pretrained(path)
        self.m = int(cfg.vision_config.spatial_merge_size)
        self.taps, self.cap = taps_for(len(self.v.blocks)), {}
        for i in self.taps:
            self.v.blocks[i].register_forward_hook(
                lambda _m, _inp, out, i=i: self.cap.__setitem__(i, out[0] if isinstance(out, (tuple, list)) else out))

    @torch.no_grad()
    def __call__(self, rgb):
        b = self.proc(images=[rgb], return_tensors="pt")
        thw = b["image_grid_thw"].cuda(); t, gh, gw = [int(v) for v in thw[0]]
        assert (t, gh * 16, gw * 16) == (1, rgb.shape[0], rgb.shape[1]), f"resized to grid {(t, gh, gw)}"
        self.cap.clear()
        self.v(b["pixel_values"].cuda().to(torch.bfloat16), grid_thw=thw)
        out = []
        for i in self.taps:
            x = self.cap[i].float().reshape(gh // self.m, gw // self.m, self.m, self.m, -1)
            x = x.permute(0, 2, 1, 3, 4).reshape(gh * gw, -1)          # undo Qwen's merge-window patch order
            out.append(F.layer_norm(x, (x.shape[-1],)))
        return torch.cat(out, -1), (gh, gw)


class Dino:
    stride = 16

    def __init__(self, mid="facebook/dinov3-vitb16-pretrain-lvd1689m", taps=(2, 5, 8, 11)):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(mid).cuda().eval()
        self.taps = taps
        self.ns = 1 + getattr(self.m.config, "num_register_tokens", 0)
        self.mean = np.array([0.485, 0.456, 0.406], np.float32); self.std = np.array([0.229, 0.224, 0.225], np.float32)

    @torch.no_grad()
    def __call__(self, rgb):
        x = torch.from_numpy(((rgb.astype(np.float32) / 255 - self.mean) / self.std).transpose(2, 0, 1)[None]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = self.m(pixel_values=x, output_hidden_states=True).hidden_states
        gh, gw = rgb.shape[0] // 16, rgb.shape[1] // 16
        return torch.cat([F.layer_norm(hs[t + 1][0, self.ns:].float(), (hs[t + 1].shape[-1],)) for t in self.taps], -1), (gh, gw)


class QwenViT:
    stride = 16

    def __init__(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        path = os.path.join(ROOT, "external", "WeVisOne-VL-2B")
        self.proc = AutoProcessor.from_pretrained(path).image_processor
        full = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0})
        self.v = full.model.visual.eval()
        full.model.language_model = None; full.lm_head = None; del full; torch.cuda.empty_cache()
        self.m = int(self.v.spatial_merge_size)
        self.taps, self.cap = (5, 11, 17, 23), {}
        for idx, mg in zip(self.v.deepstack_visual_indexes, self.v.deepstack_merger_list):
            mg.register_forward_pre_hook(lambda _m, inp, i=int(idx): self.cap.__setitem__(i, inp[0]))
        self.v.merger.register_forward_pre_hook(lambda _m, inp: self.cap.__setitem__(len(self.v.blocks) - 1, inp[0]))

    @torch.no_grad()
    def __call__(self, rgb):
        b = self.proc(images=[rgb], return_tensors="pt")
        thw = b["image_grid_thw"].cuda(); t, gh, gw = [int(v) for v in thw[0]]
        assert (t, gh * 16, gw * 16) == (1, rgb.shape[0], rgb.shape[1]), f"resized to grid {(t, gh, gw)}"
        self.cap.clear()
        self.v(b["pixel_values"].cuda().to(torch.bfloat16), grid_thw=thw)
        out = []
        for i in self.taps:
            x = self.cap[i].float().reshape(gh // self.m, gw // self.m, self.m, self.m, -1)
            x = x.permute(0, 2, 1, 3, 4).reshape(gh * gw, -1)          # undo Qwen's merge-window patch order
            out.append(F.layer_norm(x, (x.shape[-1],)))
        return torch.cat(out, -1), (gh, gw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True,
                    choices=["dinov3", "dinov3l", "dinov3h", "qwenvit", "qwen38", "cradio_h", "cradio_so"])
    ap.add_argument("--test", type=int, default=1, help="also score the 5,506 competition test strips (label-free)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0); rng = np.random.default_rng(0)
    X_ = lambda *p: os.path.join(ROOT, "external", *p)
    enc = {"dinov3": Dino, "qwenvit": QwenViT, "qwen38": Qwen38,
           "dinov3l": lambda: Dino("facebook/dinov3-vitl16-pretrain-lvd1689m", (5, 11, 17, 23)),
           "dinov3h": lambda: Dino(X_("dinov3-vith16plus"), taps_for(32)),
           "cradio_h": lambda: Radio(X_("C-RADIOv4-H")),
           "cradio_so": lambda: Radio(X_("C-RADIOv4-SO400M"))}[a.encoder]()
    tr = pd.read_csv(os.path.join(SPLIT_DIR, "train12k_ids.csv"))["ImageId"].tolist()
    np.random.default_rng(0).shuffle(tr); tr = tr[:N_POOL]
    ho = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    rd = lambda f: cv2.imread(os.path.join(DATA_DIR, "train_images", f))[:, :, ::-1].copy()

    t0 = time.time(); X, Y, pool_tr, ytr = [], [], [], []
    for k, f in enumerate(tr):
        feat, (gh, gw) = enc(rd(f)); m = mask_of(f)
        pool_tr.append(torch.cat([feat.mean(0), feat.amax(0)]).half().cpu())
        ytr.append([(m == c).any() for c in range(1, 5)])
        if k < N_PROBE:
            t = soft_targets_at_stride(m, enc.stride).reshape(5, -1).T
            bg = np.flatnonzero(t[:, 0] >= 0.999)
            keep = np.concatenate([np.flatnonzero(t[:, 0] < 0.999), rng.choice(bg, min(BG_PER_IMG, len(bg)), replace=False)])
            X.append(feat[torch.from_numpy(keep).cuda()].half().cpu()); Y.append(torch.from_numpy(t[keep]).float())
        if (k + 1) % 500 == 0:
            print(f"{a.encoder} train {k + 1}/{len(tr)} | {(time.time() - t0) / (k + 1):.2f} s/img", flush=True)
    X, Y = torch.cat(X), torch.cat(Y); D = X.shape[1]
    print(f"probe set {tuple(X.shape)} | feature dim {D}", flush=True)

    lin = nn.Linear(D, 5).cuda(); opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    for s in range(600):
        j = torch.randint(0, len(X), (8192,))
        loss = F.cross_entropy(lin(X[j].cuda().float()), Y[j].cuda()); opt.zero_grad(); loss.backward(); opt.step()

    inter, ps, gs = np.zeros(5), np.zeros(5), np.zeros(5); pool_ho, yho = [], []
    for f in ho:
        feat, (gh, gw) = enc(rd(f)); m = mask_of(f)
        with torch.no_grad():
            pred = lin(feat).argmax(-1).cpu().numpy()
        g = soft_targets_at_stride(m, enc.stride).reshape(5, -1).argmax(0)
        for c in range(1, 5):
            inter[c] += ((pred == c) & (g == c)).sum(); ps[c] += (pred == c).sum(); gs[c] += (g == c).sum()
        pool_ho.append(torch.cat([feat.mean(0), feat.amax(0)]).half().cpu()); yho.append([(m == c).any() for c in range(1, 5)])
    dice = [float(2 * inter[c] / max(ps[c] + gs[c], 1)) for c in range(1, 5)]

    P_tr, P_ho = torch.stack(pool_tr).float(), torch.stack(pool_ho).float()
    mu, sd = P_tr.mean(0), P_tr.std(0) + 1e-4
    P_tr, P_ho = ((P_tr - mu) / sd).cuda(), ((P_ho - mu) / sd).cuda()
    ytr_t = torch.tensor(np.array(ytr), dtype=torch.float32).cuda(); yho = np.array(yho)
    clf = nn.Linear(P_tr.shape[1], 4).cuda(); opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-2)
    for s in range(400):
        loss = F.binary_cross_entropy_with_logits(clf(P_tr), ytr_t); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sc = torch.sigmoid(clf(P_ho)).cpu().numpy()
    from vlm_score import auc
    res = {"encoder": a.encoder, "feature_dim": D, "probe_cells": int(len(X)), "n_probe_train": N_PROBE, "n_pool_train": N_POOL,
           "n_holdout": len(ho), "probe_dice_c1_c4": [round(v, 4) for v in dice], "probe_mean_dice": round(float(np.mean(dice)), 4),
           "presence_auc": {f"c{c + 1}": round(auc(sc[:, c], yho[:, c]), 4) for c in range(4)},
           "presence_auc_any": round(auc(sc.max(1), yho.any(1)), 4), "holdout_positives": yho.sum(0).tolist(),
           "minutes": round((time.time() - t0) / 60, 1)}
    np.save(os.path.join(OUT, f"{a.encoder}_holdout_presence.npy"), sc)
    if a.test:                       # label-free: where do encoders disagree on the real test distribution?
        te = pd.read_csv(os.path.join(ROOT, "outputs", "pseudo", "test_ids.csv")).iloc[:, 0].tolist()
        tdir = os.path.join(DATA_DIR, "test_images")
        st = []
        for k, f in enumerate(te):
            feat, _ = enc(cv2.imread(os.path.join(tdir, f))[:, :, ::-1].copy())
            p = ((torch.cat([feat.mean(0), feat.amax(0)]).float() - mu.cuda()) / sd.cuda())[None]
            with torch.no_grad():
                st.append(torch.sigmoid(clf(p))[0].cpu().numpy())
            if (k + 1) % 1000 == 0:
                print(f"{a.encoder} test {k + 1}/{len(te)}", flush=True)
        np.save(os.path.join(OUT, f"{a.encoder}_test_presence.npy"), np.array(st))
        res["test_scored"] = len(st)
    json.dump(res, open(os.path.join(OUT, f"{a.encoder}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
