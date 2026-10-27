"""Train a lightweight segmentation head on cached frozen DINOv3 features,
with class-weighted loss to fix the imbalance collapse that made A1a predict
zero pixels for classes 1 and 2.

Two head types:
  linear : Linear(384 -> 5)                      ~1.9k params
  mlp    : Linear(384->256) + GELU + Linear(->5) ~100k params

Class weights are inverse-frequency (normalised), which is the multiclass
analogue of the `class_weight="balanced"` setting that scored 0.91-0.98
held-out-image AUC in the separability diagnostic.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn

CACHE_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\data\processed\feature_cache"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints"


def build_head(kind, dim=384, n_classes=5):
    if kind == "linear":
        return nn.Linear(dim, n_classes)
    if kind == "mlp":
        return nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, n_classes))
    raise ValueError(kind)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="linear", choices=["linear", "mlp"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=8192)
    p.add_argument("--weight-power", type=float, default=1.0,
                   help="0 = unweighted, 1 = full inverse frequency")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.load(f"{CACHE_DIR}/train_feats.npy").astype(np.float32)
    y = np.load(f"{CACHE_DIR}/train_labels.npy").astype(np.int64)
    print(f"Loaded cache: {X.shape[0]} patches, dim {X.shape[1]}")

    counts = np.array([(y == c).sum() for c in range(5)], dtype=np.float64)
    print("Per-class counts:", counts.astype(int).tolist())
    weights = (counts.sum() / np.maximum(counts, 1)) ** args.weight_power
    weights = weights / weights.mean()
    print("Class weights:", np.round(weights, 3).tolist())

    # Hold out 10% of patches purely to watch for divergence (not for
    # model selection - real selection happens on the val split later).
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y))
    n_val = len(y) // 10
    va, tr = perm[:n_val], perm[n_val:]

    Xtr = torch.from_numpy(X[tr]).to(device)
    ytr = torch.from_numpy(y[tr]).to(device)
    Xva = torch.from_numpy(X[va]).to(device)
    yva = torch.from_numpy(y[va]).to(device)

    head = build_head(args.head).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"Head '{args.head}': {n_params} trainable params\n")

    crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for ep in range(args.epochs):
        head.train()
        idx = torch.randperm(len(ytr), device=device)
        tot = 0.0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            opt.zero_grad()
            loss = crit(head(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()

        head.eval()
        with torch.no_grad():
            pred = head(Xva).argmax(1)
            accs = []
            for c in range(1, 5):
                m = yva == c
                accs.append((pred[m] == c).float().mean().item() if m.any() else float("nan"))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"epoch {ep+1:3d} | loss {tot/len(idx):.4f} | "
                  f"per-class recall c1-c4: {', '.join(f'{a:.3f}' for a in accs)}")

    path = f"{OUT_DIR}/head_{args.head}_wp{args.weight_power}.pt"
    torch.save({"state_dict": head.state_dict(), "kind": args.head}, path)
    print(f"\nSaved head to {path}")


if __name__ == "__main__":
    main()
