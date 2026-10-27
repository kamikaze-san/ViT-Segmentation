"""Zero-shot sanity check: do frozen DINOv3 patch features separate defects
from background, with no training at all?

For a handful of sample images (one per defect class), this:
  1. runs frozen DINOv3 over the image,
  2. PCA's the patch-token features down to 3 components -> RGB visualization,
  3. plots that next to the raw image and the ground-truth mask.

No training, no decoder, no loss. Pure feature inspection.
"""
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel

from rle_utils import rle_decode

DATA_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal-steel-defect-detection"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\figures"
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMG_SHAPE = (256, 1600)


def pick_sample_images(df, n_per_class=1):
    samples = []
    for cls in [1, 2, 3, 4]:
        row = df[df.ClassId == cls].iloc[0]
        samples.append((row.ImageId, cls, row.EncodedPixels))
    return samples


def get_patch_features(model, processor, image_bgr, device):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state  # (1, num_tokens, dim)

    num_register_tokens = getattr(model.config, "num_register_tokens", 0)
    num_special = 1 + num_register_tokens  # CLS + registers
    patch_tokens = last_hidden[0, num_special:, :].cpu().numpy()  # (N, dim)

    patch_size = model.config.patch_size
    h_in, w_in = inputs["pixel_values"].shape[-2:]
    grid_h, grid_w = h_in // patch_size, w_in // patch_size
    assert grid_h * grid_w == patch_tokens.shape[0], (
        f"grid mismatch: {grid_h}x{grid_w}={grid_h*grid_w} vs {patch_tokens.shape[0]} tokens"
    )
    return patch_tokens, (grid_h, grid_w)


def pca_to_rgb(patch_tokens, grid_shape):
    pca = PCA(n_components=3)
    proj = pca.fit_transform(patch_tokens)  # (N, 3)
    proj -= proj.min(axis=0)
    proj /= (proj.max(axis=0) + 1e-8)
    return proj.reshape(grid_shape[0], grid_shape[1], 3)


def main():
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    samples = pick_sample_images(df)

    print(f"Loading {MODEL_ID} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    print(f"Loaded on {device}. Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 4 * len(samples)))

    for i, (image_id, cls, rle) in enumerate(samples):
        img_path = f"{DATA_DIR}/train_images/{image_id}"
        image_bgr = cv2.imread(img_path)
        mask = rle_decode(rle, shape=IMG_SHAPE)

        patch_tokens, grid_shape = get_patch_features(model, processor, image_bgr, device)
        pca_rgb = pca_to_rgb(patch_tokens, grid_shape)
        pca_rgb_up = cv2.resize(pca_rgb, (IMG_SHAPE[1], IMG_SHAPE[0]), interpolation=cv2.INTER_NEAREST)

        axes[i, 0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f"{image_id} (class {cls}) - raw")
        axes[i, 1].imshow(pca_rgb_up)
        axes[i, 1].set_title("DINOv3 patch features (PCA->RGB, frozen, no training)")
        axes[i, 2].imshow(mask, cmap="gray")
        axes[i, 2].set_title("ground truth mask")
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    out_path = f"{OUT_DIR}/zero_shot_dinov3_probe.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
