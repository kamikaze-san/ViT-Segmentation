"""Severstal Kernels-only inference. Self-contained - paste as a single cell.

Runs locally too (--local), against our own test split, so the RLE round-trip
and the submission format can be validated BEFORE spending a submission. A
kernel that errors counts against the daily limit and returns no message, so
nothing here is left to be discovered on Kaggle.

Design notes:

  Submission columns are read from sample_submission.csv at RUNTIME, not
  hardcoded. The competition's Evaluation page documents
  `ImageId_ClassId,EncodedPixels` while the sample_submission.csv shipped with
  the data has `ImageId,EncodedPixels,ClassId`. Rather than guess, we mirror
  whatever the harness actually provides - and the re-run against the unseen
  test set supplies the authoritative template. Either way EVERY (image,class)
  pair gets a row, empty predictions included, as the spec requires
  (num_images * 4 rows).

  No HuggingFace download. The architecture comes from a local config.json and
  the weights from our fine-tuned checkpoint, so internet-off is a non-issue
  and DINOv3's gating never applies.

  fp32 by default. Kaggle GPUs are P100 (no bf16) or T4 (fp16 but not bf16),
  and every number we validated was fp32. --half enables fp16 if runtime is
  tight; at ~10-20 min for 5506 images it should not be.
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASSES = 5
GRID = (16, 100)
UPSCALE = 4
SHAPE = (256, 1600)


# ---- head definition, inlined so the kernel is one self-contained file ----
class HeadNet(nn.Module):
    def __init__(self, dim=1536, hidden=1024, n_classes=5, upscale=4):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(dim, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(hidden, n_classes * upscale * upscale, 1)
        self.shuffle = nn.PixelShuffle(upscale) if upscale > 1 else nn.Identity()

    def forward(self, x):
        return self.shuffle(self.out(self.trunk(x)))


def rle_encode(mask):
    """Kaggle RLE: 1-indexed (start, length) pairs, column-major order."""
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def load_model(model_dir, device):
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(model_dir)
    backbone = AutoModel.from_config(cfg)
    ck = torch.load(f"{model_dir}/model.pt", map_location="cpu", weights_only=False)
    missing, unexpected = backbone.load_state_dict(ck["backbone"], strict=True)
    taps = tuple(ck["tap_layers"])
    head = HeadNet(dim=384 * len(taps), hidden=ck["hidden"], upscale=UPSCALE)
    head.load_state_dict(ck["head"])
    n_special = 1 + getattr(cfg, "num_register_tokens", 0)
    print(f"loaded {ck.get('source_checkpoint')} | val {ck['val_mean_dice']:.4f} "
          f"| taps {taps} | hidden {ck['hidden']} | n_special {n_special}")
    return backbone.to(device).eval(), head.to(device).eval(), taps, n_special


@torch.no_grad()
def predict(backbone, head, taps, n_special, paths, device, norm, bs=4, half=False):
    """Yields (path, predicted label map at 256x1600) one image at a time."""
    for i0 in range(0, len(paths), bs):
        chunk = paths[i0:i0 + bs]
        imgs = []
        for pth in chunk:
            img = cv2.imread(pth)
            rgb = img[:, :, ::-1].astype(np.float32) / 255.0
            imgs.append(((rgb - norm["mean"]) / norm["std"]).transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(imgs)).to(device)
        ctx = (torch.autocast("cuda", dtype=torch.float16) if half
               else torch.autocast("cuda", enabled=False))
        with ctx:
            out = backbone(pixel_values=x, output_hidden_states=True)
            parts = []
            for i in taps:
                h = out.hidden_states[i + 1][:, n_special:, :]
                parts.append(F.layer_norm(h, (h.shape[-1],)))
            tok = torch.cat(parts, dim=-1)
            feat = tok.permute(0, 2, 1).reshape(len(chunk), -1, GRID[0], GRID[1])
            cells = head(feat).float().argmax(1)
        cells = cells.cpu().numpy().astype(np.uint8)
        for k, pth in enumerate(chunk):
            yield pth, np.repeat(np.repeat(cells[k], UPSCALE, 0), UPSCALE, 1)


def build_rows(image_ids, preds, columns):
    """One row per (image, class) pair, empty predictions included.

    `columns` comes from the harness's own sample_submission.csv so the output
    matches whichever of the two documented formats is actually in force.
    """
    rows = []
    wide = "ImageId_ClassId" in columns
    for image_id in image_ids:
        pred = preds[image_id]
        for c in (1, 2, 3, 4):
            m = (pred == c).astype(np.uint8)
            rle = rle_encode(m) if m.any() else ""
            if wide:
                rows.append({"ImageId_ClassId": f"{image_id}_{c}", "EncodedPixels": rle})
            else:
                rows.append({"ImageId": image_id, "EncodedPixels": rle, "ClassId": c})
    return pd.DataFrame(rows)[list(columns)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comp-dir", default="/kaggle/input/severstal-steel-defect-detection")
    p.add_argument("--model-dir", default="/kaggle/input/severstal-dinov3-fullft")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--half", action="store_true", help="fp16 autocast (T4-safe; NOT bf16)")
    p.add_argument("--limit", type=int, default=None, help="debug: first N images only")
    p.add_argument("--local", action="store_true",
                   help="validate against our own labelled split instead of the "
                        "competition test set, and report Dice")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    with open(f"{a.model_dir}/norm.json") as f:
        nj = json.load(f)
    norm = {"mean": np.array(nj["mean"], np.float32), "std": np.array(nj["std"], np.float32)}
    backbone, head, taps, n_special = load_model(a.model_dir, device)

    if a.local:
        split_dir = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
        img_dir = f"{a.comp_dir}/train_images"
        image_ids = pd.read_csv(f"{split_dir}/test_ids.csv")["ImageId"].tolist()
        columns = ["ImageId_ClassId", "EncodedPixels"]
    else:
        sub = pd.read_csv(f"{a.comp_dir}/sample_submission.csv")
        columns = list(sub.columns)
        img_dir = f"{a.comp_dir}/test_images"
        key = "ImageId_ClassId" if "ImageId_ClassId" in columns else "ImageId"
        image_ids = sorted({s.rsplit("_", 1)[0] if key == "ImageId_ClassId" else s
                            for s in sub[key].astype(str)})
        print(f"sample_submission columns: {columns}  -> "
              f"{'ImageId_ClassId' if 'ImageId_ClassId' in columns else 'ImageId+ClassId'} format")
    if a.limit:
        image_ids = image_ids[:a.limit]
    print(f"{len(image_ids)} images -> {len(image_ids)*4} rows expected")

    paths = [f"{img_dir}/{i}" for i in image_ids]
    preds, done = {}, 0
    for pth, pred in predict(backbone, head, taps, n_special, paths, device,
                             norm, bs=a.bs, half=a.half):
        preds[os.path.basename(pth)] = pred
        done += 1
        if done % 500 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(paths)}  {el:.0f}s elapsed, "
                  f"~{el/done*len(paths):.0f}s projected", flush=True)

    df = build_rows(image_ids, preds, columns)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(df)} rows, {(df.EncodedPixels != '').sum()} non-empty")
    print(f"total {time.time()-t0:.0f}s")

    if a.local:
        mask_dir = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\masks"
        inter = np.zeros(N_CLASSES, np.int64)
        psum = np.zeros(N_CLASSES, np.int64)
        gsum = np.zeros(N_CLASSES, np.int64)
        for image_id in image_ids:
            gt = cv2.imread(f"{mask_dir}/{os.path.splitext(image_id)[0]}.png",
                            cv2.IMREAD_UNCHANGED)
            pr = preds[image_id]
            for c in range(N_CLASSES):
                pc, gc = pr == c, gt == c
                inter[c] += np.logical_and(pc, gc).sum()
                psum[c] += pc.sum(); gsum[c] += gc.sum()
        dice = [(2 * inter[c] / (psum[c] + gsum[c])) if (psum[c] + gsum[c]) > 0 else np.nan
                for c in range(N_CLASSES)]
        print("\nLOCAL VALIDATION (should reproduce the recorded test score):")
        for c, n in enumerate(["background", "defect_1", "defect_2", "defect_3", "defect_4"]):
            print(f"  {n:>12}: {dice[c]:.4f}")
        print(f"  {'mean 1-4':>12}: {np.nanmean(dice[1:]):.4f}")


if __name__ == "__main__":
    main()
