"""DINOv3-ConvNeXt backbone + trainable multi-scale (FPN-style) segmentation head.

Why ConvNeXt (spec A3): every DINOv3 ViT is patch-16, so a ViT head is capped
at a 16x100 grid (patch-grid oracle 0.754). ConvNeXt is hierarchical and emits
stride 4/8/16/32 maps natively, so fusing at stride 4 lifts the oracle to 0.936.

Why no feature cache: caching multi-scale maps at stride 4 would need ~648 GB.
ConvNeXt-T runs at 14.6 ms/img (2.1 min/epoch over the train split) - faster
than ViT-S, since there is no quadratic attention over 1600 tokens - so the
frozen backbone is simply run inside the training loop instead.

Per-stage normalisation is NOT optional: measured raw activations differ ~11x
in mean magnitude between stages (stage 3 mean_abs 0.681 vs stage 2 0.062,
max 118.8 vs 2.9). Without it the hottest stage dominates the fusion, which is
exactly what nearly buried the multi-layer ViT experiment.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

MODEL_ID = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
STAGE_CHANNELS = (96, 192, 384, 768)   # strides 4, 8, 16, 32
BASE_STRIDE = 4
N_CLASSES = 5


def load_backbone(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    norm = {"mean": np.array(proc.image_mean, dtype=np.float32),
            "std": np.array(proc.image_std, dtype=np.float32)}
    return model, norm, device


def preprocess(images_bgr, norm, device):
    """list of HxWx3 BGR uint8 -> normalised (B,3,H,W) tensor"""
    batch = []
    for img in images_bgr:
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - norm["mean"]) / norm["std"]
        batch.append(rgb.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(batch)).to(device, non_blocking=True)


@torch.no_grad()
def backbone_features(model, x):
    """Returns the 4 stage feature maps (strides 4, 8, 16, 32)."""
    out = model(pixel_values=x, output_hidden_states=True)
    return [out.hidden_states[i] for i in range(1, 5)]


class ConvNeXtFPNHead(nn.Module):
    def __init__(self, in_chs=STAGE_CHANNELS, proj=128, hidden=128,
                 n_classes=N_CLASSES, upscale=1):
        super().__init__()
        self.upscale = upscale
        # GroupNorm(1, C) == LayerNorm over channels; equalises stage scales
        # before the learned projection sees them.
        self.norms = nn.ModuleList([nn.GroupNorm(1, c) for c in in_chs])
        self.projs = nn.ModuleList([nn.Conv2d(c, proj, 1) for c in in_chs])
        self.fuse = nn.Sequential(
            nn.Conv2d(proj * len(in_chs), hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.out = nn.Conv2d(hidden, n_classes * upscale * upscale, 1)
        self.shuffle = nn.PixelShuffle(upscale) if upscale > 1 else nn.Identity()

    def forward(self, stages):
        ref = stages[0].shape[-2:]           # stride-4 spatial size
        feats = []
        for s, nrm, prj in zip(stages, self.norms, self.projs):
            f = prj(nrm(s))
            if f.shape[-2:] != ref:
                f = F.interpolate(f, size=ref, mode="bilinear", align_corners=False)
            feats.append(f)
        return self.shuffle(self.out(self.fuse(torch.cat(feats, dim=1))))

    @property
    def output_stride(self):
        return BASE_STRIDE // self.upscale


def labels_at_stride(mask, stride):
    """Per-cell label: most frequent defect class in the cell, else background."""
    gh, gw = 256 // stride, 1600 // stride
    blocks = (mask.reshape(gh, stride, gw, stride)
                  .transpose(0, 2, 1, 3)
                  .reshape(gh * gw, stride * stride))
    counts = np.stack([(blocks == c).sum(axis=1) for c in [1, 2, 3, 4]], axis=1)
    has = counts.sum(axis=1) > 0
    lab = np.zeros(gh * gw, dtype=np.uint8)
    lab[has] = counts[has].argmax(axis=1) + 1
    return lab.reshape(gh, gw)
