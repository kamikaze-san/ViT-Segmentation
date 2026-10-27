"""Cache per-class SIGMOID probabilities for a bce_dice checkpoint.

Every existing scorer (ensemble_test, threshold_optimizer, postproc_sweep) reads
probs_{name}_{split}.npy and expects channel c to hold P(class c). For the
softmax models that came from a 5-way softmax. A bce_dice model has no
background channel and no competition between classes: P(class c) is just
sigmoid(logit_c), and the four values do not sum to 1.

Writing the sigmoid values into the same (n, 5, 64, 400) layout means the whole
downstream toolchain works unchanged - postproc_sweep only ever thresholds
channels 1..4, and never softmaxes or argmaxes. Channel 0 is left at zero: it is
the softmax's background and has no meaning here.

Optional 3-view TTA (none/hflip/vflip), matching the submitted kernel.
"""
import argparse
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoImageProcessor

from spatial_head import build, N_CLASSES
from train_lora import DATA_DIR, SPLIT_DIR, CKPT_DIR, GRID

ENS = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\ensemble"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="written to probs_{name}_{split}.npy")
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--bs", type=int, default=4)
    a = ap.parse_args()
    device = "cuda"
    os.makedirs(ENS, exist_ok=True)

    ck = torch.load(f"{CKPT_DIR}/{a.ckpt}", map_location=device, weights_only=False)
    loss = ck.get("loss", "ce_dice")
    if loss != "bce_dice":
        raise SystemExit(f"{a.ckpt} was trained with loss={loss!r}; this script is for "
                         f"bce_dice only - use ensemble_test.py for softmax models")
    model_id = ck.get("model_id")
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    model.load_state_dict(ck["backbone"])
    taps = tuple(ck["tap_layers"])
    dim = model.config.hidden_size
    head = build("conv", dim=dim * len(taps), upscale=4, hidden=ck["hidden"]).to(device).eval()
    head.load_state_dict(ck["head"])
    proc = AutoImageProcessor.from_pretrained(model_id)
    mean = np.array(proc.image_mean, np.float32); std = np.array(proc.image_std, np.float32)
    ns = 1 + getattr(AutoConfig.from_pretrained(model_id), "num_register_tokens", 0)
    print(f"{a.ckpt} | {model_id} dim {dim} | taps {taps} | tta {a.tta}")

    def logits(x):
        o = model(pixel_values=x, output_hidden_states=True)
        tok = torch.cat([F.layer_norm(o.hidden_states[t + 1][:, ns:, :],
                                      (o.hidden_states[t + 1].shape[-1],)) for t in taps], -1)
        return head(tok.permute(0, 2, 1).reshape(len(x), -1, GRID[0], GRID[1])).float()

    ids = pd.read_csv(f"{SPLIT_DIR}/{a.split}_ids.csv")["ImageId"].tolist()
    out = np.lib.format.open_memmap(f"{ENS}/probs_{a.name}_{a.split}.npy", mode="w+",
                                    dtype=np.float16, shape=(len(ids), N_CLASSES, 64, 400))
    for i0 in range(0, len(ids), a.bs):
        ch = ids[i0:i0 + a.bs]
        x = np.stack([(((cv2.imread(f"{DATA_DIR}/train_images/{f}")[:, :, ::-1]
                         .astype(np.float32) / 255.0) - mean) / std).transpose(2, 0, 1) for f in ch])
        x = torch.from_numpy(x).to(device)
        views = [("none", x)]
        if a.tta:
            views += [("h", torch.flip(x, [3])), ("v", torch.flip(x, [2]))]
        acc = 0.0
        for v, xv in views:
            p = torch.sigmoid(logits(xv))
            if v == "h":
                p = torch.flip(p, [3])
            elif v == "v":
                p = torch.flip(p, [2])
            acc = acc + p
        p = (acc / len(views)).cpu().numpy().astype(np.float16)
        p[:, 0] = 0.0                      # no background channel under sigmoid
        out[i0:i0 + len(ch)] = p
        if (i0 + a.bs) % 200 < a.bs:
            print(f"    {min(i0+a.bs, len(ids))}/{len(ids)}", flush=True)
    out.flush()
    print(f"wrote probs_{a.name}_{a.split}.npy")


if __name__ == "__main__":
    main()
