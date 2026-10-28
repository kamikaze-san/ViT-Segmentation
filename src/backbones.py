"""Non-DINOv3 backbones for train_full_ft.py (--backbone cradio | qwenvit).

Each adapter wraps the encoder module (`.module`, what LoRA and the optimiser see) and returns exactly what
tap_features() returns for DINOv3: the 4 tapped layers, each LayerNorm'd per patch, concatenated, as a
(B, 4*dim, 16, 100) map at stride 16 for a 256x1600 strip. Taps sit at the same relative depths as our
validated (2, 5, 8, 11) of 12 blocks. Inputs arrive already normalised by read_batch with `self.norm`.

  cradio   NVIDIA C-RADIOv4 (SO400M or H), remote code. Wants RGB in [0, 1] (normalises itself), so norm is
           mean 0 / std 1. Features from radio_model.forward_intermediates. LoRA: qkv/proj/fc1/fc2 of the ViT blocks.
  qwenvit  Qwen3-VL vision tower (24 x 1024), loaded alone from the WeVisOne-VL-2B weights. Patchified in torch
           exactly like Qwen's image processor (2 identical temporal frames, 2x2 merge windows), mean/std 0.5.
           Taps 5/11/17/23 captured at the merger inputs and un-shuffled to row-major. LoRA: qkv/proj/linear_fc1/2.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def taps_for(n):
    return tuple(int(round((t + 1) * n / 12)) - 1 for t in (2, 5, 8, 11))


def wrap_blocks_with_checkpoint(blocks):
    """Activation checkpointing that does not depend on the model's own loop honouring a flag
    (RADIO's forward_intermediates calls each block directly)."""
    for b in blocks:
        orig = b.forward

        def fwd(*a, _orig=orig, _b=b, **k):
            if _b.training and torch.is_grad_enabled():
                return checkpoint(_orig, *a, use_reentrant=False, **k)
            return _orig(*a, **k)
        b.forward = fwd


class RadioBackbone:
    kind = "cradio"

    def __init__(self, folder, device):
        from transformers import AutoModel
        self.module = AutoModel.from_pretrained(folder, trust_remote_code=True).to(device)
        vit = self.module.radio_model.model
        self.n_blocks = len(vit.blocks)
        self.taps = taps_for(self.n_blocks)
        self.dim = int(vit.embed_dim)
        self.norm = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}
        self.lora_regex = r".*blocks\.\d+\.(attn\.qkv|attn\.proj|mlp\.fc1|mlp\.fc2)"
        self.blocks = vit.blocks
        self.pos_gen = vit.patch_generator

    def train_mode_fix(self):
        # In train mode C-RADIO's CPE patch generator randomly crops/stretches the position-embedding grid per sample
        # and drops it for 10% of samples. On 16x100 strips even the unaugmented train geometry differs from eval's
        # (rows spread over the full grid height vs its top 16%), so LoRA would train on positions eval never sees.
        self.pos_gen.eval()

    def __call__(self, x):
        outs = self.module.radio_model.forward_intermediates(x, indices=list(self.taps), output_fmt="NLC",
                                                             intermediates_only=True)
        gh, gw = x.shape[-2] // 16, x.shape[-1] // 16
        tok = torch.cat([F.layer_norm(o, (o.shape[-1],)) for o in outs], -1)
        return tok.permute(0, 2, 1).reshape(x.shape[0], -1, gh, gw)


class QwenViTBackbone:
    kind = "qwenvit"

    def __init__(self, folder, device):
        from transformers import AutoConfig, Qwen3VLVisionModel
        from safetensors import safe_open
        cfg = AutoConfig.from_pretrained(folder).vision_config
        self.module = Qwen3VLVisionModel(cfg)
        sd = {}
        with safe_open(os.path.join(folder, "model.safetensors"), "pt") as f:
            for k in f.keys():
                if k.startswith("model.visual."):
                    sd[k[len("model.visual."):]] = f.get_tensor(k)
        miss, unexp = self.module.load_state_dict(sd, strict=False)
        assert not miss, f"qwen vision weights missing: {miss[:5]}"
        self.module = self.module.to(device)
        self.ps, self.tps, self.m = int(cfg.patch_size), int(cfg.temporal_patch_size), int(cfg.spatial_merge_size)
        self.n_blocks = len(self.module.blocks)
        self.taps = (5, 11, 17, 23)
        assert set(self.taps[:-1]) <= set(int(i) for i in self.module.deepstack_visual_indexes), "taps must be deepstack layers"
        self.dim = int(cfg.hidden_size)
        self.norm = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
        self.lora_regex = r".*blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"
        self.blocks = self.module.blocks
        self.cap = {}
        for idx, mg in zip(self.module.deepstack_visual_indexes, self.module.deepstack_merger_list):
            mg.register_forward_pre_hook(lambda _m, inp, i=int(idx): self.cap.__setitem__(i, inp[0]))
        self.module.merger.register_forward_pre_hook(
            lambda _m, inp: self.cap.__setitem__(self.n_blocks - 1, inp[0]))

    def train_mode_fix(self):
        pass  # the Qwen ViT has no train-only behaviour (attention dropout is hard-coded 0)

    def patchify(self, x):
        """(B, 3, H, W) normalised -> Qwen pixel_values (B*gh*gw, 3*tps*ps*ps) and grid_thw, as the processor does."""
        B, C, H, W = x.shape
        ps, tps, m = self.ps, self.tps, self.m
        gh, gw = H // ps, W // ps
        t = x.unsqueeze(1).expand(B, tps, C, H, W)                               # 2 identical frames
        t = t.reshape(B, 1, tps, C, gh // m, m, ps, gw // m, m, ps)
        t = t.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)                              # B, gt, gh/m, gw/m, m, m, C, tps, ps, ps
        pv = t.reshape(B * gh * gw, C * tps * ps * ps)
        thw = torch.tensor([[1, gh, gw]] * B, device=x.device)
        return pv, thw, gh, gw

    def __call__(self, x):
        pv, thw, gh, gw = self.patchify(x)
        self.cap.clear()
        self.module(pv, grid_thw=thw)
        B, m = x.shape[0], self.m
        out = []
        for i in self.taps:
            h = self.cap[i].reshape(B, gh // m, gw // m, m, m, -1)
            h = h.permute(0, 1, 3, 2, 4, 5).reshape(B, gh * gw, -1)               # undo merge-window order
            out.append(F.layer_norm(h, (h.shape[-1],)))
        tok = torch.cat(out, -1)
        return tok.permute(0, 2, 1).reshape(B, -1, gh, gw)


def build_backbone(kind, folder, device):
    return {"cradio": RadioBackbone, "qwenvit": QwenViTBackbone}[kind](folder, device)
