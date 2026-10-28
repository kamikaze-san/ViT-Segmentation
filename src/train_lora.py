"""LoRA fine-tuning of DINOv3 + the multi-scale conv head.

Everything before this point kept the backbone frozen. The capacity/epoch
ablations showed frozen features plateau at ~0.7135 val: an 11x bigger head
bought +0.0197 and 2.5x more epochs bought +0.0018, against a 0.936 ceiling.
So the features themselves are the binding constraint - this adapts them.

Design choices and why:

  LoRA over partial unfreezing
      The head taps blocks 2/5/8/11. Unfreezing only the last N layers would
      leave the early taps frozen while the late ones adapt, feeding the head
      an inconsistent mix. LoRA adapts all 12 blocks at low rank so every tap
      moves together. DinoDental (arXiv 2603.28297) also reports LoRA matching
      full fine-tuning for DINOv3 segmentation at ~34x fewer trainable params.

  Head initialised from the best frozen-feature checkpoint
      LP-FT (linear-probe-then-fine-tune): starting from a random head sends
      large, uninformative gradients into pretrained weights on the first
      steps and distorts them. Our trained head already knows the task, and
      the shapes match exactly, so this costs nothing.

  Native 256x1600, no crops
      Measured 1.11 GB peak at batch 4 (SDPA never materialises the 1600x1600
      attention matrix, and LoRA keeps optimiser state tiny). Training on
      square crops and evaluating on full strips would force a large
      aspect-ratio shift through the position embeddings for no benefit.

  Separate LR for adapters vs head
      Layer-wise LR decay is a technique for *pretrained* weights; LoRA
      parameters are freshly initialised, so a single adapter LR is the
      standard choice.

  Loss unchanged (weighted CE + soft-Dice)
      Kept identical to the frozen runs so the comparison isolates adaptation.
"""
import argparse
import os
import time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # before torch loads
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from peft import LoraConfig, get_peft_model

from spatial_head import build, patch_labels_at_stride, soft_dice_loss, N_CLASSES

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# raw competition data: train_images/, test_images/, train.csv (not in git)
DATA_DIR = os.environ.get("SEVERSTAL_DATA_DIR", f"{_REPO}/data/raw")
MASK_DIR = os.environ.get("SEVERSTAL_MASK_DIR", f"{_REPO}/data/processed/masks")
SPLIT_DIR = os.environ.get("SEVERSTAL_SPLIT_DIR", f"{_REPO}/data/processed")
CKPT_DIR = os.environ.get("SEVERSTAL_CKPT_DIR", f"{_REPO}/outputs/checkpoints")


def pick_device():
    """cuda > mps > cpu; override with SEVERSTAL_DEVICE."""
    d = os.environ.get("SEVERSTAL_DEVICE")
    if d:
        return d
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()
# bf16 everywhere by default (as on the PC); SEVERSTAL_AMP=fp16 to try half precision on MPS
AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}[os.environ.get("SEVERSTAL_AMP", "bf16")]
TAP_LAYERS = (2, 5, 8, 11)
GRID = (16, 100)


def load_backbone(rank, alpha, dropout, device):
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj"])
    model = get_peft_model(model, cfg).to(device)
    norm = {"mean": np.array(proc.image_mean, dtype=np.float32),
            "std": np.array(proc.image_std, dtype=np.float32)}
    return model, norm


def tap_features(model, x, n_special):
    """Multi-layer tap, per-layer LayerNorm'd then concatenated -> (B,1536,16,100).

    The LayerNorm is not cosmetic: raw DINOv3 activations grow ~59x in
    magnitude from block 2 to block 11, and without it the deepest tap
    dominates the concatenation.
    """
    out = model(pixel_values=x, output_hidden_states=True)
    parts = []
    for i in TAP_LAYERS:
        h = out.hidden_states[i + 1][:, n_special:, :]
        parts.append(F.layer_norm(h, (h.shape[-1],)))
    tok = torch.cat(parts, dim=-1)
    b = tok.shape[0]
    return tok.permute(0, 2, 1).reshape(b, -1, GRID[0], GRID[1])


def read_batch(ids, sel, stride, norm):
    imgs, labs, masks = [], [], []
    for i in sel:
        img = cv2.imread(f"{DATA_DIR}/train_images/{ids[i]}")
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        imgs.append(((rgb - norm["mean"]) / norm["std"]).transpose(2, 0, 1))
        m = cv2.imread(f"{MASK_DIR}/{os.path.splitext(ids[i])[0]}.png", cv2.IMREAD_UNCHANGED)
        labs.append(patch_labels_at_stride(m, stride))
        masks.append(m)
    return np.stack(imgs), np.stack(labs), masks


@torch.no_grad()
def evaluate(model, head, ids, stride, norm, device, n_special, bs=4, limit=None):
    """Pixel-level Dice, predictions upsampled to 256x1600 - same accumulation
    as evaluate.py so numbers stay comparable to every earlier run."""
    model.eval(); head.eval()
    use = ids[:limit] if limit else ids
    inter = np.zeros(N_CLASSES, np.int64)
    psum = np.zeros(N_CLASSES, np.int64)
    gsum = np.zeros(N_CLASSES, np.int64)
    for i0 in range(0, len(use), bs):
        sel = list(range(i0, min(i0 + bs, len(use))))
        x, _, masks = read_batch(use, sel, stride, norm)
        xt = torch.from_numpy(x).to(device)
        with torch.autocast(DEVICE, dtype=AMP_DTYPE):
            cells = head(tap_features(model, xt, n_special)).float().argmax(1)
        cells = cells.cpu().numpy().astype(np.uint8)
        for k in range(len(sel)):
            pred = np.repeat(np.repeat(cells[k], stride, 0), stride, 1)
            gt = masks[k]
            for c in range(N_CLASSES):
                pc, gc = pred == c, gt == c
                inter[c] += np.logical_and(pc, gc).sum()
                psum[c] += pc.sum(); gsum[c] += gc.sum()
    return np.array([(2 * inter[c] / (psum[c] + gsum[c])) if (psum[c] + gsum[c]) > 0 else np.nan
                     for c in range(N_CLASSES)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--weight-power", type=float, default=0.30)
    p.add_argument("--dice", type=float, default=1.0)
    p.add_argument("--init-head", default="spatial_conv_up4_h1024_wp0.3_dice1.0_ep30.pt")
    p.add_argument("--freeze-epochs", type=int, default=0,
                   help="keep LoRA disabled for this many epochs (head-only warmup)")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--tag", default="")
    args = p.parse_args()

    device = DEVICE
    stride = 4
    model, norm = load_backbone(args.rank, args.alpha, args.dropout, device)
    n_special = 1 + getattr(model.base_model.model.config, "num_register_tokens", 0)
    model.print_trainable_parameters()

    head = build("conv", dim=384 * len(TAP_LAYERS), upscale=4, hidden=args.hidden).to(device)
    if args.init_head:
        ck = torch.load(f"{CKPT_DIR}/{args.init_head}", map_location=device, weights_only=False)
        head.load_state_dict(ck["state_dict"])
        print(f"head initialised from {args.init_head} (val {ck.get('val_mean_dice')})")

    tr_ids = pd.read_csv(f"{SPLIT_DIR}/train_ids.csv")["ImageId"].tolist()
    va_ids = pd.read_csv(f"{SPLIT_DIR}/val_ids.csv")["ImageId"].tolist()
    if args.max_train:
        tr_ids = tr_ids[:args.max_train]

    # class weights from the same stride-4 label distribution used before
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for image_id in tr_ids:
        m = cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png", cv2.IMREAD_UNCHANGED)
        lab = patch_labels_at_stride(m, stride)
        for c in range(N_CLASSES):
            counts[c] += (lab == c).sum()
    w = (counts.sum() / np.maximum(counts, 1)) ** args.weight_power
    w = w / w.mean()
    print("class weights:", np.round(w, 3).tolist())

    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    lora_params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lora_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(0)

    base = evaluate(model, head, va_ids, stride, norm, device, n_special,
                    bs=args.bs, limit=args.val_limit)
    print(f"epoch 0 (head as-loaded, LoRA at init=identity) | val mean {np.nanmean(base[1:]):.4f}")

    best = (-1, None)
    for ep in range(args.epochs):
        model.train(); head.train()
        warm = ep < args.freeze_epochs
        for q in lora_params:
            q.requires_grad_(not warm)

        order = rng.permutation(len(tr_ids))
        tot, nb, t0 = 0.0, 0, time.time()
        for i0 in range(0, len(order), args.bs):
            sel = order[i0:i0 + args.bs]
            x, labs, _ = read_batch(tr_ids, sel, stride, norm)
            xt = torch.from_numpy(x).to(device)
            y = torch.from_numpy(labs).long().to(device)
            opt.zero_grad()
            with torch.autocast(DEVICE, dtype=AMP_DTYPE):
                logits = head(tap_features(model, xt, n_special)).float()
                loss = crit(logits, y)
                if args.dice > 0:
                    loss = loss + args.dice * soft_dice_loss(logits, y)
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        d = evaluate(model, head, va_ids, stride, norm, device, n_special,
                     bs=args.bs, limit=args.val_limit)
        m = np.nanmean(d[1:])
        flag = ""
        if m > best[0]:
            best = (m, {"lora": {k: v.detach().cpu().clone()
                                 for k, v in model.state_dict().items() if "lora" in k},
                        "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}})
            flag = "  <- best"
        print(f"ep {ep+1:2d}{'(warm)' if warm else '     '} | loss {tot/nb:.4f} | {time.time()-t0:5.0f}s | "
              f"val c1 {d[1]:.4f} c2 {d[2]:.4f} c3 {d[3]:.4f} c4 {d[4]:.4f} | mean {m:.4f}{flag}",
              flush=True)

    path = f"{CKPT_DIR}/lora_r{args.rank}_h{args.hidden}_wp{args.weight_power}{args.tag}.pt"
    torch.save({"lora": best[1]["lora"], "head": best[1]["head"], "rank": args.rank,
                "alpha": args.alpha, "hidden": args.hidden, "tap_layers": TAP_LAYERS,
                "val_mean_dice": float(best[0])}, path)
    print(f"\nBEST val mean Dice (c1-4): {best[0]:.4f}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
