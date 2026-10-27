"""Correct DINOv3 patch-feature extraction for Severstal images.

IMPORTANT: the HuggingFace AutoImageProcessor default resizes to 224x224
square, which squashes our 256x1600 images ~7x horizontally and destroys
thin defects. This module bypasses the processor's resize and feeds images
at native resolution, giving a 16x100 = 1600 patch grid instead of 14x14.
"""
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMG_SHAPE = (256, 1600)
PATCH_SIZE = 16


def load_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    norm = {"mean": np.array(proc.image_mean), "std": np.array(proc.image_std)}
    return model, norm, device


DEFAULT_LAYERS = (2, 5, 8, 11)  # 0-indexed blocks -> 3rd, 6th, 9th, 12th of 12


@torch.no_grad()
def extract_patch_features(model, norm, device, image_bgr, layers=None):
    """Returns (patch_tokens [N, D], grid_shape (gh, gw)) at native resolution.

    layers=None  -> final layer only, D = 384
    layers=tuple -> those hidden states concatenated, D = 384 * len(layers).
                    Early blocks retain edge/texture detail that the final
                    block abstracts away, which matters for thin defects.
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - norm["mean"]) / norm["std"]
    x = torch.from_numpy(rgb.transpose(2, 0, 1)[None].astype(np.float32)).to(device)

    h, w = x.shape[-2:]
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)

    if layers is None:
        out = model(pixel_values=x)
        tokens = out.last_hidden_state[0, n_special:, :]
    else:
        out = model(pixel_values=x, output_hidden_states=True)
        # hidden_states[0] is the embedding output, so block i -> index i+1.
        # Raw ViT activations grow ~59x in magnitude from block 2 to block 11,
        # so each layer is LayerNorm'd before concatenation - otherwise the
        # deepest layer dominates and the early (edge/texture) layers are
        # effectively ignored by the head.
        parts = []
        for i in layers:
            hs = out.hidden_states[i + 1][0, n_special:, :]
            parts.append(torch.nn.functional.layer_norm(hs, (hs.shape[-1],)))
        tokens = torch.cat(parts, dim=-1)

    tokens = tokens.float().cpu().numpy()
    gh, gw = h // PATCH_SIZE, w // PATCH_SIZE
    assert gh * gw == tokens.shape[0], f"grid {gh}x{gw} != {tokens.shape[0]} tokens"
    return tokens, (gh, gw)


def patch_labels_from_mask(mask, grid_shape):
    """Downsample a full-res binary mask to per-patch labels (1 if any defect
    pixel falls in the patch)."""
    gh, gw = grid_shape
    small = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return (small > 0).astype(np.uint8).reshape(-1)
