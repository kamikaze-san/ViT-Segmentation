"""Score a saved head checkpoint on the 500 holdout strips with train_full_ft.evaluate() itself (same code, same
metric as the training log), so a head trained elsewhere (e.g. the Mac's C-RADIO stage-1 head) can be checked
against its reported number before anything is built on it.

  python eval_head.py --ckpt <file in outputs/checkpoints> [--backbone dinov3|cradio|qwenvit --backbone-path <dir>]
  For dinov3 the backbone comes from the checkpoint's model_id (frozen stage-1 heads store the stock backbone).
"""
import argparse, os
import numpy as np, pandas as pd, torch

import train_full_ft as T
from spatial_head import build
from transformers import AutoModel, AutoImageProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--backbone", default=None, help="default: the checkpoint's backbone_kind, else dinov3")
    ap.add_argument("--backbone-path", default=None)
    ap.add_argument("--split", default="holdout")
    a = ap.parse_args()
    from train_lora import DEVICE
    dev = DEVICE
    ck = torch.load(os.path.join(T.CKPT_DIR, a.ckpt), map_location="cpu", weights_only=False)
    kind = a.backbone or ck.get("backbone_kind") or "dinov3"
    if kind == "dinov3":
        mid = ck.get("model_id")
        proc = AutoImageProcessor.from_pretrained(mid)
        model = AutoModel.from_pretrained(mid).to(dev).eval()
        if ck.get("backbone") is not None:
            model.load_state_dict(ck["backbone"])
        T.TAP_LAYERS = tuple(ck["tap_layers"])
        norm = {"mean": np.array(proc.image_mean, np.float32), "std": np.array(proc.image_std, np.float32)}
        n_special, dim = 1 + getattr(model.config, "num_register_tokens", 0), model.config.hidden_size
    else:
        from backbones import build_backbone
        path = a.backbone_path or ck.get("backbone_path")
        T.ADAPTER = build_backbone(kind, path, dev)
        model = T.ADAPTER.module.eval()
        if ck.get("backbone") is not None:
            model.load_state_dict(ck["backbone"])
        T.TAP_LAYERS = tuple(T.ADAPTER.taps)
        saved = tuple(ck.get("tap_layers") or ())
        assert not saved or saved == T.TAP_LAYERS, f"checkpoint taps {saved} != adapter taps {T.TAP_LAYERS}"
        norm = {"mean": np.array(T.ADAPTER.norm["mean"], np.float32), "std": np.array(T.ADAPTER.norm["std"], np.float32)}
        n_special, dim = 0, T.ADAPTER.dim
    head = build(ck.get("head_arch", "conv"), dim=dim * len(T.TAP_LAYERS), upscale=4, hidden=ck["hidden"]).to(dev)
    head.load_state_dict(ck["head"])
    ids = pd.read_csv(os.path.join(T.SPLIT_DIR, f"{a.split}_ids.csv"))["ImageId"].tolist()
    d = T.evaluate(model, head, ids, 4, norm, dev, n_special, bs=4,
                   decision="sigmoid" if ck.get("loss") == "bce_dice" else "argmax")
    print(f"{a.ckpt} [{kind}] on {a.split} ({len(ids)}): c1 {d[1]:.4f} c2 {d[2]:.4f} c3 {d[3]:.4f} c4 {d[4]:.4f} "
          f"| mean {np.nanmean(d[1:]):.4f} | reported in checkpoint {ck.get('val_mean_dice')}")


if __name__ == "__main__":
    main()
