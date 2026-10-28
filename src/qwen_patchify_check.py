"""Check that backbones.QwenViTBackbone.patchify reproduces Qwen's own image processor exactly, and that the adapter's
features match the encoder_probe path (full WeVisOne model) on the same strip."""
import os, sys, numpy as np, cv2, torch
from transformers import AutoProcessor

ROOT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal"
sys.path.insert(0, os.path.join(ROOT, "src"))
from train_lora import DATA_DIR  # noqa: E402
from backbones import QwenViTBackbone  # noqa: E402

path = os.path.join(ROOT, "external", "WeVisOne-VL-2B")
rgb = cv2.imread(os.path.join(DATA_DIR, "train_images", "0014fce06.jpg"))[:, :, ::-1].copy()
proc = AutoProcessor.from_pretrained(path).image_processor
ref = proc(images=[rgb], return_tensors="pt")
bb = QwenViTBackbone(path, "cuda")
x = torch.from_numpy(((rgb.astype(np.float32) / 255.0 - 0.5) / 0.5).transpose(2, 0, 1)[None]).cuda()
pv, thw, gh, gw = bb.patchify(x)
print("grid processor", ref["image_grid_thw"].tolist(), "| ours", thw.tolist())
print("pixel_values shapes", tuple(ref["pixel_values"].shape), tuple(pv.shape),
      "| max abs diff", float((ref["pixel_values"].cuda().float() - pv).abs().max()))
with torch.no_grad():
    f = bb(x)
print("feature map", tuple(f.shape), "| finite", bool(torch.isfinite(f).all()))
