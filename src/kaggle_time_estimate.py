"""How long would a solo Kaggle run of a new bundle take, relative to one whose Kaggle time we know?

Times pure inference (model loading excluded) of two bundles on the same test images, the way kaggle_ensemble_cell_v2
computes it: fp16 autocast, 3-view TTA, softmax, then per class a bilinear resize to 256x1600 and threshold. Kaggle's
time scales roughly with the ratio (an upper bound: loading and the CPU-side RLE do not scale with the model).
  python kaggle_time_estimate.py [--n 100]
"""
import argparse, os, time
import numpy as np, cv2, torch, torch.nn.functional as F

from train_lora import DATA_DIR
from backbones import build_backbone
from spatial_head import build

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "..", "outputs")
BUNDLES = {"soft_vitb": os.path.join(B, "kaggle_bundle_calib", "model_vitb_lora_warm_soft_pseudo01.pt"),
           "cradio": os.path.join(B, "kaggle_bundle_cradio", "model_cradio_so_soft_pseudo01.pt")}
TTA = ("none", "h", "v")
IMAGENET = (torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))


def remap(sd, want):                                  # as the cell: bundles may or may not carry a "model." prefix
    if set(sd) == want:
        return sd
    s = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    return s if set(s) == want else {("model." + k): v for k, v in sd.items()}


def load(name, dev):
    ck = torch.load(BUNDLES[name], map_location="cpu", weights_only=False, mmap=True)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in ck["backbone"].items()}
    if ck.get("backbone_kind") == "cradio":
        ad = build_backbone("cradio", os.path.join(HERE, "..", "external", "C-RADIOv4-SO400M"), dev)
        ad.module.load_state_dict(sd, strict=True); ad.module.eval()
        feats, dim, norm = ad, ad.dim, (torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1))
    else:
        from transformers import AutoModel
        bb = AutoModel.from_pretrained(ck.get("model_id") or "facebook/dinov3-vitb16-pretrain-lvd1689m").to(dev).eval()
        bb.load_state_dict(remap(sd, set(bb.state_dict())), strict=True)
        taps, ns = tuple(ck["tap_layers"]), 1 + getattr(bb.config, "num_register_tokens", 0)

        def feats(x):
            o = bb(pixel_values=x, output_hidden_states=True)
            tok = torch.cat([F.layer_norm(o.hidden_states[t + 1][:, ns:], (o.hidden_states[t + 1].shape[-1],)) for t in taps], -1)
            return tok.permute(0, 2, 1).reshape(len(x), -1, 16, 100)
        dim, norm = bb.config.hidden_size, IMAGENET
    hd = build(ck.get("head_arch", "conv"), dim=dim * len(ck["tap_layers"]), upscale=4, hidden=ck["hidden"])
    hd.load_state_dict({k: v.float() for k, v in ck["head"].items()}); hd = hd.to(dev).eval()
    return feats, hd, (norm[0].to(dev), norm[1].to(dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    dev = "cuda"
    files = sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))[:a.n]
    imgs = [(cv2.imread(os.path.join(DATA_DIR, "test_images", f))[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)
            for f in files]
    res = {}
    for name in BUNDLES:
        feats, hd, (mu, sd) = load(name, dev)

        def run(chunk):
            x = (torch.from_numpy(np.stack(chunk)).to(dev) - mu) / sd
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                acc = 0.0
                for v in TTA:
                    xv = x if v == "none" else torch.flip(x, [3] if v == "h" else [2])
                    p = hd(feats(xv)).float().softmax(1)
                    acc = acc + (p if v == "none" else torch.flip(p, [3] if v == "h" else [2]))
                p = (acc / len(TTA)).cpu().numpy()
            for k in range(len(chunk)):
                for c in range(1, 5):
                    _ = cv2.resize(p[k, c], (1600, 256), interpolation=cv2.INTER_LINEAR) > 0.5
        run(imgs[:4]); torch.cuda.synchronize()                      # warm-up
        t0 = time.time()
        for i in range(0, len(imgs), 4):
            run(imgs[i:i + 4])
        torch.cuda.synchronize()
        res[name] = (time.time() - t0) / len(imgs)
        print(f"{name}: {res[name]:.4f} s/img (n={len(imgs)}, fp16, 3-view TTA)", flush=True)
        del feats, hd
        torch.cuda.empty_cache()
    r = res["cradio"] / res["soft_vitb"]
    print(f"ratio cradio / soft_vitb = {r:.2f}x | if the soft ViT-B solo took T min on Kaggle, C-RADIO ~ {r:.2f} x T "
          f"(e.g. T = 14 -> ~{14 * r:.0f} min, an upper bound)", flush=True)


if __name__ == "__main__":
    main()
