"""Full fine-tuning of DINOv3 + the multi-scale conv head.

This is the decisive test of the "feature adaptation is saturated" claim.
The LoRA rank sweep (r=4/8/16 -> val 0.7016/0.7013/0.7030) was read as
saturation, but that reading had two holes:

  1. ViT-S has d=384, so r=4/8/16 spans only 1%-4% of full rank. A flat
     response inside a narrow low-rank band is not evidence of a ceiling.
  2. LoRA targeted q_proj/k_proj/v_proj ONLY. The MLP blocks (up_proj,
     down_proj) hold ~2/3 of each block's parameters and were never
     adapted at all. For domain shift that is often where the movement
     needs to happen.

Full fine-tuning removes both limits at once - every module, full rank.
If this still lands at ~0.70, supervised adaptation really is exhausted
and the residual gap to the 0.936 stride-4 oracle is about labels, not
about backbone capacity.

Design choices and why:

  Layer-wise LR decay, mild (gamma=0.85 default, not the 0.75 usually
  quoted for ViT-S/16 classification)
      LLRD protects early pretrained features from large early gradients.
      But our head taps blocks 2/5/8/11, so it consumes early features
      directly - at gamma=0.75 block 2 would get 0.75^10 = 5.6% of base LR
      and effectively stay frozen, which would quietly re-impose the very
      constraint this run exists to remove. 0.85 gives block 2 ~20%.

  Head initialised from the best frozen-feature checkpoint (LP-FT)
      Same as the LoRA runs. At step 0 the backbone is bit-identical to
      the frozen one the head was trained on, so this is a clean
      continuation rather than a restart, and the first gradients into the
      pretrained weights are informative instead of random.

  No weight decay on norms, biases, layer_scale, or the special tokens
      Decaying 1-D parameters shrinks LayerNorm gains and the layer_scale
      lambdas toward zero, which attenuates residual branches. This
      matters far more in full FT than in LoRA, where those params were
      frozen anyway.

  Gradient clipping at 1.0
      LoRA is intrinsically damped (zero-init B, alpha/r scaling). Full FT
      has no such damping and a single bad batch can move pretrained
      weights irrecoverably.

  Everything else identical to train_lora.py
      Same loss (weighted CE + soft-Dice), same weight_power, same native
      256x1600 input, same bf16 autocast, same pixel-level evaluate().
      The comparison is meant to isolate adaptation capacity.
"""
import argparse
import copy
import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

from spatial_head import build, patch_labels_at_stride, soft_dice_loss, N_CLASSES
from train_lora import (MODEL_ID, DATA_DIR, MASK_DIR, SPLIT_DIR, CKPT_DIR, TAP_LAYERS,
                        GRID, read_batch)

N_BLOCKS = 12
MAX_DEPTH = N_BLOCKS + 1  # embeddings=0, blocks=1..12, final norm=13


def grid_distort(img, mask, rng, steps=(4, 10), limit=0.08):
    """Mild non-rigid warp that PRESERVES image size.

    Rotation and scaling are still excluded - they change the 256x1600 input
    and therefore the 16x100 patch grid. Grid distortion perturbs cell
    boundaries and resamples, so the output is exactly the same size. The mask
    is warped by the identical map with nearest-neighbour interpolation so
    class ids are never blended.
    """
    h, w = img.shape[:2]
    ys, xs = steps
    x_edges = np.linspace(0, w, xs + 1)
    y_edges = np.linspace(0, h, ys + 1)
    x_new = x_edges + rng.uniform(-limit, limit, xs + 1) * (w / xs)
    y_new = y_edges + rng.uniform(-limit, limit, ys + 1) * (h / ys)
    x_new[0], x_new[-1], y_new[0], y_new[-1] = 0, w, 0, h
    mx = np.tile(np.interp(np.arange(w), x_new, x_edges).astype(np.float32), (h, 1))
    my = np.repeat(np.interp(np.arange(h), y_new, y_edges).astype(np.float32)[:, None], w, 1)
    return (cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101),
            cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101))


def augment_pair(img, mask, rng, strength="mild"):
    """Shape-preserving augmentation only.

    Rotation and scaling are deliberately excluded: they would change the
    256x1600 input size and therefore the 16x100 patch grid the head is
    built around, and would need padding that invents steel texture at the
    borders. Flips and photometric jitter leave the geometry exact.

    Flips apply to image AND mask; photometric ops to the image only.

    The 'blackout*' strengths layer defect_blackout ON TOP of 'strong', so the
    only variable versus the strong-augmentation runs is the blackout itself.

    'mild' is the exact recipe B9 was trained with, kept reproducible.
    'strong' widens the appearance ops and adds gamma, sensor noise, blur and
    a mild non-rigid warp. The emphasis is deliberately on APPEARANCE rather
    than geometry: the Kaggle result showed we OVER-PREDICT on the hidden set
    rather than mislocalise, which points at illumination/sensor differences
    between production runs, not at geometric variation.
    """
    if strength.startswith("blackout") or strength == "copypaste":
        strength = "strong"
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        img, mask = img[::-1], mask[::-1]

    if strength == "mild":
        if rng.random() < 0.5:
            alpha, beta = 1.0 + rng.uniform(-0.2, 0.2), rng.uniform(-20.0, 20.0)
            img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)

    img = np.ascontiguousarray(img)
    mask = np.ascontiguousarray(mask)
    if rng.random() < 0.5:                       # wider brightness / contrast
        alpha, beta = 1.0 + rng.uniform(-0.3, 0.3), rng.uniform(-35.0, 35.0)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:                       # gamma: nonlinear tone response,
        g = rng.uniform(0.7, 1.4)                # which linear contrast cannot emulate
        img = np.clip(((img / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:                       # sensor / exposure noise
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(3, 15), img.shape),
                      0, 255).astype(np.uint8)
    if rng.random() < 0.2:                       # focus and line-speed variation
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.0))
    if rng.random() < 0.3:
        img, mask = grid_distort(img, mask, rng)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def _fill_from_clean(img, mask, y0, y1, x0, x1, rng, tile=48):
    """Paint (y0:y1, x0:x1) with defect-free steel from the same image.

    Tries one donor the size of the region; if none is clean enough (large
    defects leave little clean area), tiles smaller donors instead. Returns
    False only if even a tile-sized clean patch cannot be found.
    """
    h, w = mask.shape
    bh, bw = y1 - y0, x1 - x0
    for _ in range(12):
        sy, sx = rng.integers(0, h - bh + 1), rng.integers(0, w - bw + 1)
        if not mask[sy:sy + bh, sx:sx + bw].any():
            img[y0:y1, x0:x1] = img[sy:sy + bh, sx:sx + bw]
            return True
    filled_any = False
    for ty in range(y0, y1, tile):
        for tx in range(x0, x1, tile):
            th, tw = min(tile, y1 - ty), min(tile, x1 - tx)
            for _ in range(8):
                sy, sx = rng.integers(0, h - th + 1), rng.integers(0, w - tw + 1)
                if not mask[sy:sy + th, sx:sx + tw].any():
                    img[ty:ty + th, tx:tx + tw] = img[sy:sy + th, sx:sx + tw]
                    filled_any = True
                    break
    return filled_any


def defect_blackout(img, mask, rng, p_apply=0.4, fill="steel"):
    """1st place's blackout, with the artifact removed.

    Their version zeroes the defect's bounding box and flips the image's label
    to 0 when every defect is erased. The mechanism is to break CONTEXTUAL
    shortcuts: without it a model can learn "images that look like this run tend
    to have defects" and fire on global appearance rather than on the defect
    actually being visible. That is precisely our failure - we over-predict on
    unseen steel. They report it "works especially on class 1", our weakest class.

    Their fill is literal black, which is a perfectly label-correlated artifact:
    every black box sits exactly where a defect was removed, so the model can
    learn "black box -> nothing here" instead of learning what clean steel looks
    like. That shortcut is worthless at test time. The author flagged it himself:
    "it may be better to fill it with nondefective region values".

    So we fill with a same-sized patch copied from a DEFECT-FREE region of the
    SAME image - same lighting, same sensor, same texture - leaving nothing for
    the model to latch onto. The context stays, the evidence goes.
    """
    if rng.random() > p_apply or not mask.any():
        return img, mask
    h, w = mask.shape
    boxes = []
    for c in range(1, N_CLASSES):
        ys, xs = np.where(mask == c)
        if len(ys) == 0:
            continue
        boxes.append((c, ys.min(), ys.max() + 1, xs.min(), xs.max() + 1))
    if not boxes:
        return img, mask
    rng.shuffle(boxes)
    n_erase = rng.integers(1, len(boxes) + 1)          # sometimes all -> image becomes negative
    img, mask = img.copy(), mask.copy()
    for c, y0, y1, x0, x1 in boxes[:n_erase]:
        bh, bw = y1 - y0, x1 - x0
        if bh >= h or bw >= w:
            continue
        if fill == "black":
            img[y0:y1, x0:x1] = 0
        else:
            # Try one clean donor the size of the whole box; large c3 defects can
            # span most of the strip, leaving no clean region that big, so fall
            # back to tiling smaller clean patches. Without the fallback ~half of
            # defective images were left untouched and only small defects were
            # ever erased.
            if not _fill_from_clean(img, mask, y0, y1, x0, x1, rng):
                continue
        mask[y0:y1, x0:x1] = 0                          # label follows the pixels
    return img, mask


PSEUDO_PREFIX = "pseudo:"     # train-list ids with this prefix are pseudo-labelled test images
PSEUDO_MASK_SUBDIR = "pseudo_masks"   # under SPLIT_DIR; set from --pseudo-mask-dir (round-1 labels by default)


def img_mask_paths(image_id):
    """Real training images live in train_images with human masks; pseudo-labelled
    competition test images live in test_images with the teacher masks written by
    build_pseudo_labels.py. One resolver, so the two sources can never be crossed."""
    if image_id.startswith(PSEUDO_PREFIX):
        f = image_id[len(PSEUDO_PREFIX):]
        return (f"{DATA_DIR}/test_images/{f}",
                f"{SPLIT_DIR}/{PSEUDO_MASK_SUBDIR}/{os.path.splitext(f)[0]}.png")
    return (f"{DATA_DIR}/train_images/{image_id}",
            f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png")


def read_batch_aug(ids, sel, stride, norm, rng, strength="mild"):
    """read_batch + augmentation. Labels are derived AFTER augmenting so the
    patch-majority assignment matches the warped/flipped mask."""
    raw = []
    for i in sel:
        img_path, mask_path = img_mask_paths(ids[i])
        img = cv2.imread(img_path)
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        raw.append(augment_pair(img, m, rng, strength))

    imgs, labs, masks = [], [], []
    for k, (img, m) in enumerate(raw):
        if strength in ("blackout", "blackout_black", "copypaste"):
            img, m = defect_blackout(img, m, rng,
                                     fill="black" if strength == "blackout_black" else "steel")
        if strength == "copypaste" and len(raw) > 1:
            # donor is another image from the SAME batch - no extra disk read, and
            # same-batch images already went through identical photometric jitter
            j = (k + 1 + int(rng.integers(len(raw) - 1))) % len(raw)
            img, m = copy_paste(img, m, raw[j][0], raw[j][1], rng)
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        imgs.append(((rgb - norm["mean"]) / norm["std"]).transpose(2, 0, 1))
        labs.append(patch_labels_at_stride(m, stride))
        masks.append(m)
    return np.stack(imgs), np.stack(labs), masks


def copy_paste(img, mask, donor_img, donor_mask, rng, p_apply=0.4):
    """Paste a real defect from a donor image into this one.

    Blackout removes evidence while keeping context; copy-paste does the inverse -
    puts evidence into a context that never had it. Both attack the same shortcut
    (firing on global appearance rather than on a visible defect), from opposite
    directions.

    Two choices matter for the artifact, which is the reason this was deprioritised
    earlier: the seam sits exactly where the positive label is, so anything the
    model can latch onto there is a perfectly label-correlated cue.

      - paste the defect's ACTUAL MASK SHAPE, not its bounding box. A bbox paste
        drops a rectangle of donor steel around the defect, and the rectangle edge
        is a straight line no real defect produces.
      - match local brightness using the BACKGROUND either side, not the defect.
        Matching on the defect pixels themselves shifts them to the destination's
        background level, which erases the defect entirely and trains the model on
        positive labels over untouched steel. Referencing the non-defect pixels
        inside the same bounding box preserves the defect's contrast against its
        surroundings while removing the illumination step.
    """
    if rng.random() > p_apply or not donor_mask.any():
        return img, mask
    present = [c for c in (1, 2, 3, 4) if (donor_mask == c).any()]
    if not present:
        return img, mask
    c = int(present[rng.integers(len(present))])
    sel = donor_mask == c
    ys, xs = np.where(sel)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = donor_img[y0:y1, x0:x1].astype(np.float32)
    pm = sel[y0:y1, x0:x1]
    h, w = pm.shape
    H, W = img.shape[:2]
    if h > H or w > W or pm.sum() < 16:
        return img, mask
    dy = int(rng.integers(0, H - h + 1))
    dx = int(rng.integers(0, W - w + 1))
    dst = img[dy:dy + h, dx:dx + w].astype(np.float32)
    # brightness match on the BACKGROUND inside the bbox, never on the defect
    bg = ~pm
    shift = (dst[bg].mean() - patch[bg].mean()) if bg.any() else 0.0
    patch = np.clip(patch + shift, 0, 255)
    dst[pm] = patch[pm]
    img[dy:dy + h, dx:dx + w] = dst.astype(np.uint8)
    mask[dy:dy + h, dx:dx + w][pm] = c
    return img, mask


def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension of the Jaccard loss (Berman et al.)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(logits, labels, classes="present"):
    """Lovasz-softmax: a convex surrogate that optimises IoU DIRECTLY, rather
    than optimising pixel likelihood and hoping IoU follows.

    5th and 9th place both switched to Lovasz after training with a pixel loss.
    9th place's reasoning: with cross-entropy "the loss is decreasing but the
    metrics is not increasing". 5th place reports it "helped the model to filter
    out false-positive masks" - false positives being 55% of our loss.

    'present' restricts the sum to classes actually in the batch, which matters
    here because class 2 appears in 2% of images and would otherwise contribute
    a constant zero-gradient term.
    """
    probas = logits.softmax(1)
    B, C = probas.shape[:2]
    probas = probas.permute(0, 2, 3, 1).reshape(-1, C)
    labels = labels.reshape(-1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg[perm])))
    return sum(losses) / max(len(losses), 1)


IS_CONVNEXT = False      # set from the loaded config in main()


def convnext_stages(model, x):
    """The 4 hierarchical stage maps (strides 4, 8, 16, 32).

    ConvNeXt is not a ViT: there are no patch tokens to reshape, and it emits a
    stride-4 map natively rather than needing PixelShuffle from stride 16. The
    FPN head fuses the stages back to stride 4, which is the same 64x400 output
    the ViT path produces - so everything downstream (losses, eval, thresholds,
    checkpointing) is untouched.
    """
    out = model(pixel_values=x, output_hidden_states=True)
    return [out.hidden_states[i] for i in range(1, 5)]


def backbone_state_for_save(model, lora_r):
    """State dict that a plain AutoModel can load.

    Under LoRA, model.state_dict() has peft-prefixed keys plus lora_A/lora_B
    tensors; AutoModel.load_state_dict would reject it outright. Merging on a
    deep copy folds W + (alpha/r) B@A back into ordinary weight matrices, so the
    saved checkpoint is architecturally a normal model and every existing
    inference path (ensemble_test, the Kaggle cell) reads it unchanged. The copy
    keeps the live training model's adapters intact.
    """
    if not lora_r:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    merged = copy.deepcopy(model).merge_and_unload()
    return {k: v.detach().cpu().clone() for k, v in merged.state_dict().items()}


def bce_dice_loss(logits, y, pos_weight, dice_w, eps=1.0):
    """Per-class INDEPENDENT sigmoid + BCE, instead of a 5-way softmax.

    This is a decision-structure change, not a fitting change. The Kaggle metric
    scores every (ImageId, ClassId) pair on its own, so a c3 prediction is never
    penalised by a c1 prediction on the same pixel - yet a softmax forces the
    classes to compete for probability mass anyway. Sigmoid matches the metric's
    own independence; a pixel may belong to two classes, exactly as the
    submission format allows.

    Channel 0 of the head is left unsupervised here (it is the softmax's
    background) so the architecture and checkpoint shapes stay identical.
    """
    tgt = torch.stack([(y == c).float() for c in range(1, N_CLASSES)], 1)
    lg = logits[:, 1:]
    loss = F.binary_cross_entropy_with_logits(lg, tgt, pos_weight=pos_weight)
    if dice_w > 0:
        p = torch.sigmoid(lg)
        dims = (0, 2, 3)
        inter = (p * tgt).sum(dims)
        den = p.sum(dims) + tgt.sum(dims)
        loss = loss + dice_w * (1.0 - ((2 * inter + eps) / (den + eps)).mean())
    return loss


def param_depth(name):
    """Map a parameter name to its depth in the network for LLRD."""
    if name.startswith("embeddings."):
        return 0
    if name.startswith("model.layer."):
        return int(name.split(".")[2]) + 1
    return MAX_DEPTH  # final norm


def no_decay(name, param):
    """Norms, biases, layer_scale lambdas and the special tokens."""
    return param.ndim <= 1 or name.endswith(("cls_token", "mask_token", "register_tokens"))


def build_param_groups(model, base_lr, gamma, wd):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (param_depth(name), no_decay(name, p))
        groups.setdefault(key, []).append(p)
    out = []
    for (d, nd), params in sorted(groups.items()):
        out.append({"params": params, "lr": base_lr * (gamma ** (MAX_DEPTH - d)),
                    "weight_decay": 0.0 if nd else wd})
    return out


def tap_features(model, x, n_special):
    if IS_CONVNEXT:
        return convnext_stages(model, x)      # the FPN head consumes the list
    out = model(pixel_values=x, output_hidden_states=True)
    parts = []
    for i in TAP_LAYERS:
        h = out.hidden_states[i + 1][:, n_special:, :]
        parts.append(F.layer_norm(h, (h.shape[-1],)))
    tok = torch.cat(parts, dim=-1)
    return tok.permute(0, 2, 1).reshape(tok.shape[0], -1, GRID[0], GRID[1])


@torch.no_grad()
def evaluate(model, head, ids, stride, norm, device, n_special, bs=4, limit=None,
             decision="argmax"):
    """Pixel-level Dice, identical accumulation to train_lora.evaluate()."""
    model.eval(); head.eval()
    use = ids[:limit] if limit else ids
    inter = np.zeros(N_CLASSES, np.int64)
    psum = np.zeros(N_CLASSES, np.int64)
    gsum = np.zeros(N_CLASSES, np.int64)
    for i0 in range(0, len(use), bs):
        sel = list(range(i0, min(i0 + bs, len(use))))
        x, _, masks = read_batch(use, sel, stride, norm)
        xt = torch.from_numpy(x).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = head(tap_features(model, xt, n_special)).float()
            if decision == "sigmoid":
                # classes decided independently; background is "nothing fired",
                # and a pixel may count for two classes - as the metric allows
                fired = torch.sigmoid(lg[:, 1:]) > 0.5
                binm = torch.cat([~fired.any(1, keepdim=True), fired], 1)
            else:
                binm = F.one_hot(lg.argmax(1), N_CLASSES).permute(0, 3, 1, 2).bool()
        binm = binm.cpu().numpy()
        for k in range(len(sel)):
            gt = masks[k]
            for c in range(N_CLASSES):
                pc = np.repeat(np.repeat(binm[k, c], stride, 0), stride, 1)
                gc = gt == c
                inter[c] += np.logical_and(pc, gc).sum()
                psum[c] += pc.sum(); gsum[c] += gc.sum()
    return np.array([(2 * inter[c] / (psum[c] + gsum[c])) if (psum[c] + gsum[c]) > 0 else np.nan
                     for c in range(N_CLASSES)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bb-lr", type=float, default=5e-5, help="LR at the TOP of the backbone")
    p.add_argument("--gamma", type=float, default=0.85, help="layer-wise LR decay")
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--head-arch", default="conv", choices=["conv", "progressive"],
                   help="conv = the shipped head: 1x1 -> 3x3 -> 1x1 emitting n_classes*16 "
                        "channels, PixelShuffle(4) to stride 4 in ONE step, so nothing "
                        "computes spatially after the rearrangement. progressive = "
                        "16->8->4 with a 3x3 conv AFTER each x2 upsample. Match params when "
                        "comparing: conv hidden 1024 = 11.09M, progressive hidden 832 = 11.19M.")
    p.add_argument("--taps", type=int, nargs="+", default=None,
                   help="which blocks the head reads, 0-indexed. Default (2,5,8,11) was "
                        "chosen for 12-block ViT-S/B. A 24-block ViT-L needs the "
                        "proportional equivalent (5 11 17 23) or the head sees only the "
                        "first half of the network.")
    p.add_argument("--model-id", default=MODEL_ID,
                   help="backbone to fine-tune. ViT-B/16 is a drop-in: it is also 12 "
                        "blocks, so TAP_LAYERS and the LLRD depth map carry over "
                        "unchanged and only the head's input dim moves 384->768. "
                        "ViT-L is 24 blocks and would need both rethought.")
    p.add_argument("--weight-power", type=float, default=0.30)
    p.add_argument("--pos-weight-power", type=float, default=0.15,
                   help="bce_dice only: pos_weight = (neg/pos) ** this, per class. "
                        "Inheriting --weight-power 0.30 gives c2 a pos_weight of 73, "
                        "which pays the model heavily for firing - directly against the "
                        "one thing we have measured about this problem (we over-predict "
                        "on the target: 9.2%% false positives there vs 0.9%% in "
                        "distribution). 0.15 halves the exponent, giving c2 ~8.5.")
    p.add_argument("--dice", type=float, default=1.0)
    p.add_argument("--loss", default="ce_dice", choices=["ce_dice", "lovasz", "ce_dice_lovasz", "bce_dice"],
                   help="ce_dice = the weighted-CE + soft-Dice used for every run so far. "
                        "bce_dice = per-class independent sigmoid+BCE, matching the metric's own "
                        "per-(image,class) independence. "
                        "lovasz = Lovasz-softmax alone, the second phase 5th/9th place used after "
                        "a pixel loss. ce_dice_lovasz = both summed.")
    p.add_argument("--lovasz-w", type=float, default=1.0)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--init-head", default="spatial_conv_up4_h1024_wp0.3_dice1.0_ep30.pt")
    p.add_argument("--resume", default=None,
                   help="continue from a full-FT checkpoint: loads BACKBONE + HEAD (and optimizer "
                        "state if the checkpoint has it). Overrides --init-head.")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--augment", action="store_true",
                   help="augment TRAIN only (eval always stays clean)")
    p.add_argument("--aug-strength", default="mild",
                   choices=["mild", "strong", "blackout", "blackout_black", "copypaste"],
                   help="mild = the exact B9 recipe; strong = wider appearance ops "
                        "+ gamma/noise/blur/grid-distortion")
    p.add_argument("--seed", type=int, default=0,
                   help="controls data-ordering (and augmentation) randomness. NOTE: full FT has no "
                        "random init at all - head comes from a checkpoint, backbone from pretrained "
                        "weights - so seed variance here is shuffle-order variance, not init variance.")
    p.add_argument("--lora-r", type=int, default=0,
                   help="LoRA rank; 0 = full fine-tune. The optimizer state (grads + two "
                        "AdamW moments) is the DOMINANT memory term and is independent of "
                        "batch size - measured: ViT-L peaks at 7599 MiB at bs=4 and 7765 "
                        "at bs=2. LoRA is what actually removes it: ~4.8 GB -> ~1.25 GB.")
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-targets", nargs="+",
                   default=["q_proj", "k_proj", "v_proj"],
                   help="which Linear modules get adapters. The historical setting adapts "
                        "only q/k/v = 3 of 6 per block, ~25%% of block weights, and never "
                        "touches the MLP (up_proj/down_proj) which holds two thirds of the "
                        "parameters. That is the likely reason the r4->r16 sweep was flat "
                        "while full FT beat LoRA by 0.022: coverage was the binding "
                        "constraint, not rank.")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="recompute activations during backward instead of storing them. "
                        "Cuts activation memory ~10x for ~30%% more compute - the "
                        "difference between ViT-L fitting at bs=4 and not fitting at all.")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="linear-probe mode: train ONLY the head on frozen features. Used to "
                        "measure whether MIM pretraining improved the representation, before "
                        "spending hours on a full fine-tune that would re-adapt it anyway")
    p.add_argument("--backbone-ckpt", default=None,
                   help="load the backbone from a MIM checkpoint instead of stock DINOv3")
    p.add_argument("--train-split", default="train",
                   help="basename of <split>_ids.csv to train on (e.g. train12k)")
    p.add_argument("--pseudo-ids", default=None,
                   help="csv of pseudo-labelled competition TEST images (build_pseudo_labels.py "
                        "--write) appended to the train split. Class weights are still computed "
                        "from the real split alone, so against the same recipe without this "
                        "flag the only thing that changes is the data.")
    p.add_argument("--pseudo-mask-dir", default="pseudo_masks",
                   help="folder under data/processed holding the masks for --pseudo-ids. Each label "
                        "set gets its own folder so one teacher's labels can never overwrite another's.")
    p.add_argument("--val-split", default="val",
                   help="basename of <split>_ids.csv to evaluate on. When training on "
                        "train12k this MUST be the holdout, since val is inside the "
                        "training set and would report a meaningless score")
    p.add_argument("--save-epochs", default="",
                   help="comma-separated epochs to snapshot regardless of val, e.g. 8,12,16. "
                        "Lets the epoch-count question be settled by measurement afterwards "
                        "instead of guessed in advance")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    device = "cuda"
    stride = 4
    # a BCE model must be SCORED the way it is trained, or epoch selection picks
    # the best softmax epoch of a model that is not doing softmax
    DECISION = "sigmoid" if args.loss == "bce_dice" else "argmax"

    proc = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device)
    global IS_CONVNEXT
    IS_CONVNEXT = "convnext" in model.config.model_type
    if IS_CONVNEXT:
        print(f"ConvNeXt backbone: hierarchical stages {model.config.hidden_sizes}, "
              f"native stride 4 - no taps, no PixelShuffle")
    # LLRD scales each block's LR by gamma ** (MAX_DEPTH - depth). With MAX_DEPTH
    # pinned at 13 and a 24-block backbone, param_depth returns up to 25 and deep
    # blocks get gamma ** -12 = ~142x the base LR - divergence, not a small error.
    # So derive both the depth map and the taps from the backbone actually loaded.
    global N_BLOCKS, MAX_DEPTH, TAP_LAYERS
    if IS_CONVNEXT:
        N_BLOCKS = MAX_DEPTH = 0
    else:
        N_BLOCKS = model.config.num_hidden_layers
    if not IS_CONVNEXT:
        MAX_DEPTH = N_BLOCKS + 1
    if args.taps and not IS_CONVNEXT:
        TAP_LAYERS = tuple(args.taps)
    elif not IS_CONVNEXT and N_BLOCKS != 12:
        # proportional to the validated 12-block choice (2,5,8,11)
        TAP_LAYERS = tuple(int(round((t + 1) * N_BLOCKS / 12)) - 1 for t in (2, 5, 8, 11))
        print(f"taps rescaled for {N_BLOCKS} blocks: {TAP_LAYERS}")
    if not IS_CONVNEXT:
        assert max(TAP_LAYERS) < N_BLOCKS, f"taps {TAP_LAYERS} exceed {N_BLOCKS} blocks"
        print(f"depth map: {N_BLOCKS} blocks, MAX_DEPTH {MAX_DEPTH}, taps {TAP_LAYERS}")
    if args.backbone_ckpt:
        bck = torch.load(f"{CKPT_DIR}/{args.backbone_ckpt}", map_location=device,
                         weights_only=False)
        model.load_state_dict(bck["backbone"])
        print(f"backbone loaded from {args.backbone_ckpt} "
              f"(MIM epoch {bck.get('epoch')}, cos {bck.get('cos')})")
    if args.grad_checkpoint and not args.freeze_backbone:
        model.gradient_checkpointing_enable()
        print("gradient checkpointing ON")
    for q in model.parameters():
        q.requires_grad_(not args.freeze_backbone)
    if args.lora_r:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
            target_modules=list(args.lora_targets))).to(device)
        n_lora = sum(q.numel() for q in model.parameters() if q.requires_grad)
        print(f"LoRA r={args.lora_r} alpha={args.lora_alpha} targets={args.lora_targets} "
              f"| {n_lora:,} trainable backbone params")
    norm = {"mean": np.array(proc.image_mean, dtype=np.float32),
            "std": np.array(proc.image_std, dtype=np.float32)}
    n_special = 1 + getattr(model.config, "num_register_tokens", 0)
    n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    mode = ("LoRA" if args.lora_r else "frozen" if args.freeze_backbone else "full fine-tune")
    print(f"backbone trainable params: {n_train:,} ({mode})")

    if IS_CONVNEXT:
        from convnext_head import ConvNeXtFPNHead
        # proj/hidden kept small: the fuse convs run at stride 4 on 64x400, so
        # hidden=1024 would be enormous. 128/256 is close to the validated A3 head.
        head = ConvNeXtFPNHead(in_chs=tuple(model.config.hidden_sizes), proj=128,
                               hidden=256, n_classes=N_CLASSES, upscale=1).to(device)
        print(f"backbone {args.model_id}  stages {tuple(model.config.hidden_sizes)}  "
              f"FPN head proj128 hidden256")
    else:
        bb_dim = model.config.hidden_size
        head = build(args.head_arch, dim=bb_dim * len(TAP_LAYERS), upscale=4,
                     hidden=args.hidden).to(device)
        print(f"backbone {args.model_id}  dim {bb_dim}  head input {bb_dim * len(TAP_LAYERS)}")
    resume_ck = None
    if args.resume:
        resume_ck = torch.load(f"{CKPT_DIR}/{args.resume}", map_location=device, weights_only=False)
        model.load_state_dict(resume_ck["backbone"])
        head.load_state_dict(resume_ck["head"])
        print(f"RESUMED backbone+head from {args.resume} (val {resume_ck['val_mean_dice']:.4f})")
        # the augmentation recipe is NOT stored in the weights, so a resume that
        # forgets --augment/--aug-strength silently continues under a different
        # data distribution than the run it is extending
        prev_aug = resume_ck.get("aug_strength")
        now_aug = args.aug_strength if args.augment else None
        if prev_aug != now_aug:
            print(f"  WARNING: checkpoint was trained with augment={prev_aug!r} but this "
                  f"run uses {now_aug!r}. Pass --augment --aug-strength {prev_aug} to match.")
        else:
            print(f"  augmentation matches the source run: {now_aug!r}")
        if "optimizer" not in resume_ck:
            print("  NOTE: checkpoint has no optimizer state - AdamW moments restart from zero. "
                  "Expect a transient dip over the first few hundred steps.")
    elif args.init_head:
        ck = torch.load(f"{CKPT_DIR}/{args.init_head}", map_location=device, weights_only=False)
        # the default warm-start head was trained on ViT-S features (1536 in). A
        # different backbone changes the head's input dim, so silently skip
        # rather than crash - the probe is then a true cold start, which is what
        # it should be when comparing backbones anyway.
        hsd = ck["state_dict"] if "state_dict" in ck else ck["head"]
        # Compare the FULL key set and shapes, not one hardcoded key: HeadNet has
        # trunk.0.weight and ProgressiveHead does not, so keying on trunk.0 raises
        # KeyError instead of reporting an incompatible head.
        cur = head.state_dict()
        want = (set(hsd) == set(cur)
                and all(hsd[k].shape == cur[k].shape for k in cur))
        got = ck.get("head_arch", "conv")
        if want:
            head.load_state_dict(hsd)
            print(f"head initialised from {args.init_head} (val {ck.get('val_mean_dice')})")
        else:
            print(f"head init SKIPPED: {args.init_head} (arch {got!r}) is not shape-"
                  f"compatible with this head - randomly initialised instead")
    else:
        print("head randomly initialised (true probe - no LP-FT warm start)")

    tr_ids = pd.read_csv(f"{SPLIT_DIR}/{args.train_split}_ids.csv")["ImageId"].tolist()
    va_ids = pd.read_csv(f"{SPLIT_DIR}/{args.val_split}_ids.csv")["ImageId"].tolist()
    overlap = set(tr_ids) & set(va_ids)
    if overlap:
        raise SystemExit(f"REFUSING TO RUN: {len(overlap)} images are in BOTH "
                         f"{args.train_split} and {args.val_split} - the eval score "
                         f"would be measured on training data")
    print(f"train split '{args.train_split}' {len(tr_ids)} | eval split "
          f"'{args.val_split}' {len(va_ids)} | no overlap")
    if args.max_train:
        tr_ids = tr_ids[:args.max_train]

    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for image_id in tr_ids:
        m = cv2.imread(f"{MASK_DIR}/{os.path.splitext(image_id)[0]}.png", cv2.IMREAD_UNCHANGED)
        lab = patch_labels_at_stride(m, stride)
        for c in range(N_CLASSES):
            counts[c] += (lab == c).sum()
    w = (counts.sum() / np.maximum(counts, 1)) ** args.weight_power
    w = w / w.mean()
    print("class weights:", np.round(w, 3).tolist())

    if args.pseudo_ids:
        # appended AFTER the class weights so the loss weighting is identical to the
        # labelled-only baseline - the pseudo set is c1/c2-poor by construction
        # (agreement selection keeps easy classes) and would otherwise shift them
        assert args.augment, "--pseudo-ids is wired into read_batch_aug only; pass --augment"
        global PSEUDO_MASK_SUBDIR
        PSEUDO_MASK_SUBDIR = args.pseudo_mask_dir
        ps = pd.read_csv(args.pseudo_ids)["ImageId"].tolist()
        assert not set(ps) & set(va_ids), "pseudo ids overlap the eval split"
        missing = [f for f in ps if not os.path.exists(img_mask_paths(PSEUDO_PREFIX + f)[1])]
        assert not missing, f"{len(missing)} pseudo masks missing, e.g. {missing[:3]}"
        tr_ids = tr_ids + [PSEUDO_PREFIX + f for f in ps]
        print(f"pseudo-labelled test images appended: {len(ps)} -> {len(tr_ids)} train ids "
              f"| masks {PSEUDO_MASK_SUBDIR}")

    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    # pos_weight for BCE, derived from the SAME counts and tempered by the same
    # --weight-power as the CE weights, rather than an invented constant. Raw
    # neg/pos runs into the hundreds for the rare classes and destabilises BCE.
    pos_w = None
    if args.loss == "bce_dice":
        pw = ((counts.sum() - counts[1:]) / np.maximum(counts[1:], 1)) ** args.pos_weight_power
        pos_w = torch.tensor(pw, dtype=torch.float32, device=device).view(-1, 1, 1)
        print("bce pos_weight:", np.round(pw, 3).tolist())
    if IS_CONVNEXT and not args.freeze_backbone and not args.lora_r:
        groups = [{"params": [q for q in model.parameters() if q.requires_grad],
                   "lr": args.bb_lr, "weight_decay": args.wd}]
        print(f"ConvNeXt: flat backbone LR {args.bb_lr:.1e} (the LLRD depth map is "
              f"ViT-block specific and does not apply)")
    elif args.lora_r:
        groups = [{"params": [q for q in model.parameters() if q.requires_grad],
                   "lr": args.bb_lr, "weight_decay": args.wd}]
        print(f"LoRA: single LR {args.bb_lr:.1e} on adapters (LLRD does not apply - the "
              f"adapters are not the pretrained blocks)")
    elif args.freeze_backbone:
        model.eval()
        groups = []
        print("FROZEN BACKBONE - training the head only (linear-probe comparison)")
    else:
        groups = build_param_groups(model, args.bb_lr, args.gamma, args.wd)
    if groups:
        print(f"LLRD: {len(groups)} backbone groups, top {args.bb_lr:.2e} -> "
              f"block0 {args.bb_lr * args.gamma ** 12:.2e} "
              f"(block2 tap {args.bb_lr * args.gamma ** 10:.2e})")
    groups.append({"params": list(head.parameters()), "lr": args.head_lr,
                   "weight_decay": args.wd})
    opt = torch.optim.AdamW(groups)
    if resume_ck is not None and "optimizer" in resume_ck:
        opt.load_state_dict(resume_ck["optimizer"])
        # the resumed state carries the OLD per-group LRs; this run's schedule owns them now
        for grp, spec in zip(opt.param_groups, groups):
            grp["lr"] = grp["initial_lr"] = spec["lr"]
        print("  optimizer state restored (AdamW moments carried over)")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)
    # separate stream so --augment does not perturb the epoch shuffle order,
    # keeping the data ordering identical to the un-augmented baseline
    aug_rng = np.random.default_rng(1234 + args.seed)

    base = evaluate(model, head, va_ids, stride, norm, device, n_special,
                    bs=args.bs, limit=args.val_limit)
    print(f"epoch 0 (head as-loaded, backbone untouched) | val mean {np.nanmean(base[1:]):.4f}",
          flush=True)

    path = f"{CKPT_DIR}/fullft_lr{args.bb_lr}_g{args.gamma}_h{args.hidden}{args.tag}.pt"

    # a resume whose output path collides with its own source would destroy the
    # checkpoint it is reading from, mid-run, with no way back
    if args.resume and os.path.abspath(path) == os.path.abspath(f"{CKPT_DIR}/{args.resume}"):
        raise SystemExit(f"REFUSING TO RUN: output path is the resume source: {path}"
                         f" -- change --tag or --bb-lr so the extension writes somewhere new")
    if os.path.exists(path):
        print(f"NOTE: {os.path.basename(path)} exists and will be overwritten once this "
              f"run beats its first epoch")

    def save_best(state, score, epoch):
        """Write on every improvement, not once at the end.

        An earlier continuation run was interrupted at epoch 15 of 16 and lost
        everything, including a best-scoring epoch-10 checkpoint, because the
        only torch.save sat after the loop. A few seconds of write per improving
        epoch is worth not losing hours of compute to an interruption.
        """
        torch.save({"backbone": state["backbone"], "head": state["head"],
                    "optimizer": state["optimizer"], "best_epoch": epoch,
                    "hidden": args.hidden, "head_arch": args.head_arch, "tap_layers": TAP_LAYERS,
                        "model_id": args.model_id, "loss": args.loss, "bb_lr": args.bb_lr,
                    "gamma": args.gamma, "augment": args.augment, "epochs": args.epochs,
                    "aug_strength": args.aug_strength if args.augment else None,
                    "seed": args.seed, "resumed_from": args.resume,
                    "pseudo_ids": args.pseudo_ids, "n_train": len(tr_ids),
                    "val_mean_dice": float(score)}, path)

    snap_at = {int(e) for e in args.save_epochs.split(",") if e.strip()}
    if snap_at:
        print(f"will snapshot at epochs {sorted(snap_at)} regardless of eval score")

    best = (-1, None)
    for ep in range(args.epochs):
        head.train()
        if not args.freeze_backbone:
            model.train()
        order = rng.permutation(len(tr_ids))
        tot, nb, t0 = 0.0, 0, time.time()
        for i0 in range(0, len(order), args.bs):
            sel = order[i0:i0 + args.bs]
            x, labs, _ = (read_batch_aug(tr_ids, sel, stride, norm, aug_rng, args.aug_strength)
                          if args.augment else read_batch(tr_ids, sel, stride, norm))
            xt = torch.from_numpy(x).to(device)
            y = torch.from_numpy(labs).long().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(tap_features(model, xt, n_special)).float()
                if args.loss == "lovasz":
                    loss = args.lovasz_w * lovasz_softmax(logits, y)
                elif args.loss == "bce_dice":
                    loss = bce_dice_loss(logits, y, pos_w, args.dice)
                else:
                    loss = crit(logits, y)
                    if args.dice > 0:
                        loss = loss + args.dice * soft_dice_loss(logits, y)
                    if args.loss == "ce_dice_lovasz":
                        loss = loss + args.lovasz_w * lovasz_softmax(logits, y)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(head.parameters()), args.clip)
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()

        d = evaluate(model, head, va_ids, stride, norm, device, n_special,
                     decision=DECISION,
                     bs=args.bs, limit=args.val_limit)
        m = np.nanmean(d[1:])
        flag = ""
        if m > best[0]:
            # optimizer state is saved WITH the best weights so a later --resume
            # continues from a consistent (weights, moments) pair rather than
            # restarting AdamW from zero the way this run had to
            best = (m, {"backbone": backbone_state_for_save(model, args.lora_r),
                        "head": {k: v.detach().cpu().clone()
                                 for k, v in head.state_dict().items()},
                        "optimizer": copy.deepcopy(opt.state_dict()),
                        "epoch": ep + 1})
            save_best(best[1], m, ep + 1)
            flag = "  <- best (saved)"
        if (ep + 1) in snap_at:
            snap = f"{CKPT_DIR}/fullft_lr{args.bb_lr}_g{args.gamma}_h{args.hidden}{args.tag}_ep{ep+1}.pt"
            torch.save({"backbone": backbone_state_for_save(model, args.lora_r),
                        "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                        "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                        "lora_targets": list(args.lora_targets) if args.lora_r else None,
                        "hidden": args.hidden, "head_arch": args.head_arch, "tap_layers": TAP_LAYERS,
                        "model_id": args.model_id, "loss": args.loss, "bb_lr": args.bb_lr,
                        "gamma": args.gamma, "augment": args.augment, "epochs": args.epochs,
                        "aug_strength": args.aug_strength if args.augment else None,
                        "seed": args.seed, "snapshot_epoch": ep + 1,
                        "train_split": args.train_split, "pseudo_ids": args.pseudo_ids,
                        "n_train": len(tr_ids), "val_mean_dice": float(m)}, snap)
            flag += f"  [snapshot ep{ep+1}]"
        print(f"ep {ep+1:2d} | loss {tot/nb:.4f} | {time.time()-t0:5.0f}s | "
              f"val c1 {d[1]:.4f} c2 {d[2]:.4f} c3 {d[3]:.4f} c4 {d[4]:.4f} | mean {m:.4f}{flag}",
              flush=True)

    # NOTE: no torch.save here. save_best() already wrote this exact state the
    # moment it became best. A second save at the end is redundant, and the
    # earlier one silently dropped the "seed" field by overwriting with a
    # different dict literal - two save paths for one artifact is a bug source.
    print(f"\nBEST val mean Dice (c1-4): {best[0]:.4f}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
