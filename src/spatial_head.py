"""Conv / sub-patch segmentation heads over cached DINOv3 spatial feature maps.

Increment A: ConvHead(upscale=1)  - 3x3 convs give each patch a ~5x5 patch
             (80x80 px) receptive field. Output stays at stride 16.
             Isolates the effect of SPATIAL CONTEXT.

Increment B: ConvHead(upscale=4)  - same trunk, but the final conv emits
             5 * 4^2 channels which PixelShuffle rearranges into a stride-4
             prediction. Escapes the one-label-per-16px-patch bound entirely
             (patch-grid oracle rises 0.754 -> 0.936).
             Isolates the effect of OUTPUT RESOLUTION.
"""
import numpy as np
import torch
import torch.nn as nn

GRID = (16, 100)
N_CLASSES = 5


class HeadNet(nn.Module):
    """Shared trunk shape so the ablation isolates ONE variable at a time.

    All variants are: 1x1 projection -> [middle layer] -> 1x1 output.
    Only the middle layer's kernel differs:

      mlp       : no middle layer          (reproduces the B3 per-patch head)
      mlp_deep  : middle is 1x1            (depth control - extra layer, NO spatial mixing)
      conv      : middle is 3x3            (adds SPATIAL CONTEXT; vs mlp_deep this is
                                            the only difference)

    Comparing conv against mlp_deep - not against mlp - is what actually
    isolates spatial context from raw added capacity.
    """

    def __init__(self, dim=1536, hidden=256, n_classes=N_CLASSES, upscale=1, middle=None):
        super().__init__()
        self.upscale = upscale
        layers = [nn.Conv2d(dim, hidden, 1), nn.GELU()]
        if middle == "1x1":
            layers += [nn.Conv2d(hidden, hidden, 1), nn.GELU()]
        elif middle == "3x3":
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU()]
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Conv2d(hidden, n_classes * upscale * upscale, 1)
        self.shuffle = nn.PixelShuffle(upscale) if upscale > 1 else nn.Identity()

    def forward(self, x):
        """x: (B, dim, gh, gw) -> (B, n_classes, gh*upscale, gw*upscale)"""
        return self.shuffle(self.out(self.trunk(x)))


class ProgressiveHead(nn.Module):
    """Stride 16 -> 8 -> 4 with convolutions at each resolution.

    HeadNet reaches stride 4 in ONE PixelShuffle: a 1x1 emits n_classes*16
    channels and they are rearranged. Every output pixel inside a 4x4 block is
    therefore produced by the same 1x1 projection of the same stride-16 feature,
    with no spatial computation after the rearrangement. That is why the boundary
    is a 4px staircase, and why interpolating the probability map afterwards
    (bilinear, +0.0005 on target) helped at all.

    This upsamples in two x2 steps and runs a 3x3 conv AFTER each one, so there
    is real spatial computation at stride 8 and stride 4 rather than only at
    stride 16.

    A UNet would also skip-connect to high-resolution encoder features here. A ViT
    has none - all four taps are stride 16 - so no amount of decoder depth can
    recover high-frequency detail the features do not carry. This tests only
    whether computation at finer strides helps; feeding raw-image detail into the
    decoder is the separate, harder experiment.
    """

    def __init__(self, dim=1536, hidden=512, n_classes=N_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        h2 = hidden // 2
        self.to8 = nn.Conv2d(hidden, h2 * 4, 1)          # -> PixelShuffle(2) -> h2 @ stride 8
        self.at8 = nn.Sequential(nn.Conv2d(h2, h2, 3, padding=1), nn.GELU())
        h4 = h2 // 2
        self.to4 = nn.Conv2d(h2, h4 * 4, 1)              # -> PixelShuffle(2) -> h4 @ stride 4
        self.at4 = nn.Sequential(nn.Conv2d(h4, h4, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(h4, n_classes, 1)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.stem(x)
        x = self.at8(self.up(self.to8(x)))
        x = self.at4(self.up(self.to4(x)))
        return self.out(x)


class SimplexHead(nn.Module):
    """HeadNet with its learned classifier replaced by a FIXED simplex ETF.

    Same trunk as HeadNet conv (1x1 -> GELU -> 3x3 -> GELU). The output 1x1 emits an
    etf_dim-dimensional embedding per stride-4 sub-pixel (etf_dim * 16 channels,
    rearranged by PixelShuffle). Each embedding is L2-normalised and the class
    logits are its cosine with K fixed unit vectors that are maximally and equally
    separated (pairwise cosine -1/(K-1), -0.25 for K=5), times a learnable scale.

    The vertices are a buffer and never trained, so rare classes cannot lose
    classifier norm to frequent ones - the imbalance mechanism of Yang et al.
    (NeurIPS 2022, arXiv 2203.09081). Output logits have exactly HeadNet shape, so
    CE + Dice, argmax, softmax and every downstream threshold path are unchanged.
    """

    def __init__(self, dim=1536, hidden=1024, n_classes=N_CLASSES, upscale=4, etf_dim=16,
                 init_scale=10.0, seed=0):
        super().__init__()
        assert etf_dim >= n_classes, "this ETF construction needs etf_dim >= n_classes"
        self.trunk = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(hidden, etf_dim * upscale * upscale, 1)
        self.shuffle = nn.PixelShuffle(upscale) if upscale > 1 else nn.Identity()
        g = torch.Generator().manual_seed(seed)
        u, _ = torch.linalg.qr(torch.randn(etf_dim, n_classes, generator=g))   # orthonormal columns
        k = n_classes
        etf = (k / (k - 1)) ** 0.5 * u @ (torch.eye(k) - torch.ones(k, k) / k)  # (etf_dim, K), unit columns
        self.register_buffer("etf", etf)
        self.log_scale = nn.Parameter(torch.tensor(float(np.log(init_scale))))

    def forward(self, x):
        """x: (B, dim, gh, gw) -> logits (B, n_classes, gh*upscale, gw*upscale)"""
        z = torch.nn.functional.normalize(self.shuffle(self.out(self.trunk(x))).float(), dim=1)
        scale = self.log_scale.float().exp().clamp(1.0, 50.0)
        return scale * torch.einsum("bdhw,dk->bkhw", z, self.etf.float())


def build(kind, dim=1536, upscale=1, hidden=256):
    if kind == "progressive":
        return ProgressiveHead(dim=dim, hidden=hidden)
    if kind == "simplex":
        return SimplexHead(dim=dim, hidden=hidden, upscale=upscale)
    middle = {"mlp": None, "mlp_deep": "1x1", "conv": "3x3"}
    if kind not in middle:
        raise ValueError(kind)
    return HeadNet(dim=dim, upscale=upscale, hidden=hidden, middle=middle[kind])


def patch_labels_at_stride(mask, stride):
    """Per-cell class label at a given pixel stride: most frequent defect
    class present in the cell, else background."""
    gh, gw = 256 // stride, 1600 // stride
    blocks = (mask.reshape(gh, stride, gw, stride)
                  .transpose(0, 2, 1, 3)
                  .reshape(gh * gw, stride * stride))
    counts = np.stack([(blocks == c).sum(axis=1) for c in [1, 2, 3, 4]], axis=1)
    has = counts.sum(axis=1) > 0
    lab = np.zeros(gh * gw, dtype=np.uint8)
    lab[has] = counts[has].argmax(axis=1) + 1
    return lab.reshape(gh, gw)


def soft_targets_at_stride(mask, stride):
    """Per-cell class FRACTIONS at a given pixel stride, shape (N_CLASSES, gh, gw).

    patch_labels_at_stride marks a cell as defect if ANY of its stride^2 pixels is,
    which inflates the training target by 6-23% in area, thin c1/c2 most (E24).
    Fractions are unbiased: summed over cells x stride^2 they equal the label's area
    exactly, and a model trained on them predicts coverage, so its edges land
    between cell centres instead of on the outer cell boundary.
    """
    gh, gw = 256 // stride, 1600 // stride
    blocks = mask.reshape(gh, stride, gw, stride).transpose(0, 2, 1, 3).reshape(gh, gw, stride * stride)
    return np.stack([(blocks == c).mean(-1) for c in range(N_CLASSES)]).astype(np.float32)


def soft_dice_loss(logits, target, n_classes=N_CLASSES, eps=1.0):
    """Per-image soft Dice, averaged over classes present in either
    prediction or target. Complements CE by optimising overlap directly."""
    probs = torch.softmax(logits, dim=1)
    oh = torch.zeros_like(probs).scatter_(1, target.unsqueeze(1), 1.0)
    dims = (0, 2, 3)
    inter = (probs * oh).sum(dims)
    denom = probs.sum(dims) + oh.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()
