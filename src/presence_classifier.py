"""Step 3+4: train a per-(image,class) presence classifier and judge it honestly.

The segmentation model ALREADY makes a presence decision - "did any pixel of
class c win the argmax?" - it just makes it implicitly, as a side effect of a
loss that never optimised it. That implicit classifier is the incumbent, and it
is what a dedicated head has to beat. This script measures both on the same
pairs so the comparison is direct.

Why an image-level head should win, mechanistically: global pooling aggregates
evidence over 1600 patches, so a handful of spurious patches barely moves it,
whereas the same handful flips a per-pixel argmax and creates a false positive.
The granularity, not the features, is the operative difference - which is also
why this runs on the FINE-TUNED backbone rather than needing a second one.

The veto threshold is derived, not tuned: emitting scores q*d in expectation and
staying empty scores 1-q, so emit iff q > 1/(1+d) where d is that class's mean
Dice on its correct detections. Vetoing a false positive gains a full 1.0 while
vetoing a true detection loses only d, so the threshold sits ABOVE 0.5.
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PRES = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\presence"
SPLIT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed"
CLASSES = [1, 2, 3, 4]


def load(split, stock=False):
    s = "_stock" if stock else ""
    a = np.load(f"{PRES}/areas_{split}{s}.npz", allow_pickle=True)
    x = np.load(f"{PRES}/pooled_{split}{s}.npz", allow_pickle=True)
    return a["P"], a["G"], a["I"], x["X"]


class Head(nn.Module):
    def __init__(self, dim, hidden=512, n_out=4, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


def fit(Xtr, Ytr, Xva, Yva, device, epochs=60, lr=1e-3, wd=1e-2, hidden=512):
    pos = Ytr.sum(0)
    pw = torch.tensor((len(Ytr) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=device)
    print(f"  pos_weight: {[f'{v:.1f}' for v in pw.tolist()]}")
    model = Head(Xtr.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=device)
    xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    best, bstate = -1, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            crit(model(xt[idx]), yt[idx]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(xv)).cpu().numpy()
        ap = np.mean([average_precision(Yva[:, k], pv[:, k]) for k in range(4)])
        if ap > best:
            best, bstate = ap, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(bstate)
    print(f"  best val mAP {best:.4f}")
    return model


def average_precision(y, s):
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def temperature(model, X, Y, device):
    """Calibrate so the derived thresholds mean what they say."""
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    T = np.ones(4)
    for k in range(4):
        best, bt = 1e9, 1.0
        for t in np.linspace(0.5, 5.0, 91):
            p = 1 / (1 + np.exp(-logits[:, k] / t))
            nll = -np.mean(Y[:, k] * np.log(p + 1e-9) + (1 - Y[:, k]) * np.log(1 - p + 1e-9))
            if nll < best:
                best, bt = nll, t
        T[k] = bt
    return logits, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", action="store_true")
    a = ap.parse_args()
    device = "cuda"

    Ptr, Gtr, Itr, Xtr = load("train", a.stock)
    Pva, Gva, Iva, Xva = load("val", a.stock)
    Ytr = (Gtr[:, 1:] > 0).astype(np.float32)
    Yva = (Gva[:, 1:] > 0).astype(np.float32)
    print(f"train {Xtr.shape} | val {Xva.shape}")
    print(f"presence rate train: {[f'{v*100:.1f}%' for v in Ytr.mean(0)]}")

    model = fit(Xtr, Ytr, Xva, Yva, device)
    logits, T = temperature(model, Xva, Yva, device)
    print(f"  temperatures: {[f'{t:.2f}' for t in T]}")
    q = 1 / (1 + np.exp(-logits / T))

    # per-class d = mean Dice on the pairs the segmentation got right
    print(f"\n{'cls':>4}{'d':>8}{'q*':>7}   incumbent(implicit)      dedicated classifier")
    print(f"{'':>4}{'':>8}{'':>7}   {'FP':>5}{'miss':>6}{'hit':>6}   {'FP':>6}{'miss':>6}{'hit':>6}")
    tot = {"impl_fp": 0, "impl_miss": 0, "cls_fp": 0, "cls_miss": 0}
    for j, c in enumerate(CLASSES):
        p, g, i = Pva[:, c], Gva[:, c], Iva[:, c]
        hit = (g > 0) & (p > 0)
        d = float((2 * i[hit] / (p[hit] + g[hit])).mean())
        qs = 1 / (1 + d)
        emit = q[:, j] > qs                       # dedicated classifier's decision
        impl = p > 0                              # the incumbent's decision
        row = (f"{c:>4}{d:>8.4f}{qs:>7.3f}   "
               f"{int(((g==0)&impl).sum()):>5}{int(((g>0)&~impl).sum()):>6}{int(((g>0)&impl).sum()):>6}   "
               f"{int(((g==0)&emit).sum()):>6}{int(((g>0)&~emit).sum()):>6}{int(((g>0)&emit).sum()):>6}")
        print(row)
        tot["impl_fp"] += int(((g == 0) & impl).sum()); tot["impl_miss"] += int(((g > 0) & ~impl).sum())
        tot["cls_fp"] += int(((g == 0) & emit).sum()); tot["cls_miss"] += int(((g > 0) & ~emit).sum())
    print(f"\n  TOTAL  implicit: FP {tot['impl_fp']}  miss {tot['impl_miss']}")
    print(f"         dedicated: FP {tot['cls_fp']}  miss {tot['cls_miss']}")

    # the gating question: of the pairs the segmentation got WRONG, how many does
    # the classifier disagree with? and does it wrongly veto correct detections?
    print("\n  GATING EFFECT on val (classifier applied on top of segmentation):")
    veto_fp = veto_hit = 0
    for j, c in enumerate(CLASSES):
        p, g = Pva[:, c], Gva[:, c]
        d_hit = (g > 0) & (p > 0)
        qs = 1 / (1 + float((2 * Iva[d_hit, c] / (p[d_hit] + g[d_hit])).mean()))
        veto = q[:, j] <= qs
        veto_fp += int(((g == 0) & (p > 0) & veto).sum())
        veto_hit += int((d_hit & veto).sum())
    print(f"    false positives vetoed : {veto_fp} of {tot['impl_fp']}"
          f"  ({100*veto_fp/max(tot['impl_fp'],1):.1f}%)")
    print(f"    true detections vetoed : {veto_hit}  (each costs ~its Dice)")
    print(f"    break-even needs FP-veto% > {100*veto_hit*0.74/max(tot['impl_fp'],1):.1f}%")

    torch.save({"state_dict": model.state_dict(), "dim": Xtr.shape[1],
                "temperature": T, "stock": a.stock}, f"{PRES}/presence_head.pt")
    print(f"\nsaved {PRES}/presence_head.pt")


if __name__ == "__main__":
    main()
