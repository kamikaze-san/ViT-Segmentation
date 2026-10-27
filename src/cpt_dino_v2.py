"""Continued pretraining of DINOv3 with its own objective family (E27).

DINO (image level) + iBOT (masked patches) + KoLeo, which is what DINOv3's backbones
were trained/distilled with, plus Gram anchoring to a FROZEN copy of stock DINOv3 -
Meta's own tool against patch-feature drift, here anchored to the starting weights
because drift is exactly how MIM failed in C2 (catastrophic forgetting found only
after 9 hours). The released checkpoints carry no DINO/iBOT heads, so heads start
random and are warmed up on a FROZEN backbone first; only then does the backbone
move, at a low LR with layer-wise decay.

Data: train12k + the 5,506 competition test images, unlabelled. Holdout is held out
so the per-epoch checks stay clean. Views at native pixel scale (no resizing): two
overlapping 224x224 global crops and eight 96x96 local crops near them, each with
its own photometric jitter and flips.

Guards, every epoch:
  drift   mean cosine between current and stock features on the four layers the
          segmentation head reads (blocks 2/5/8/11), 200 holdout images
  probe   a linear patch classifier on those features, trained on 600 train images
          and scored on 200 holdout images, against the same probe on stock
  abort   if the probe falls more than --probe-tol below stock, or any tap's
          cosine drops below --min-cos

Outputs CKPT_DIR/cpt_<tag>_ep<N>.pt with "backbone" = the EMA teacher's weights in
Hugging Face layout, so train_full_ft.py --backbone-ckpt loads it directly.

Modes: zero (save at step 0, reload, must reproduce stock exactly), smoke (tiny run
on 200 images), time (seconds per training step), train.
"""
import argparse, copy, json, math, os, time
import numpy as np, pandas as pd, cv2, torch
import torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel

from spatial_head import soft_targets_at_stride
from train_lora import DATA_DIR, MASK_DIR, SPLIT_DIR, CKPT_DIR

OUT = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\cpt"
TAPS = (2, 5, 8, 11)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
G, L, NG, NL = 224, 96, 2, 8


# ---------------------------------------------------------------- data
def photometric(x, rng):
    x = x.astype(np.float32)
    if rng.random() < 0.8:
        x = x * (1 + rng.uniform(-0.3, 0.3)) + rng.uniform(-30, 30)
    if rng.random() < 0.3:
        x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.2:
        x = cv2.GaussianBlur(x, (5, 5), rng.uniform(0.1, 1.5))
    if rng.random() < 0.2:
        x = x + rng.normal(0, rng.uniform(2, 10), x.shape)
    x = np.clip(x, 0, 255)
    if rng.random() < 0.5:
        x = x[:, ::-1]
    if rng.random() < 0.5:
        x = x[::-1]
    return ((x / 255.0 - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)


class Views(torch.utils.data.Dataset):
    def __init__(self, paths, seed=0):
        self.paths, self.seed = paths, seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, int(time.time() * 1e6) % 2**31))
        img = cv2.imread(self.paths[i])[:, :, ::-1]
        h, w = img.shape[:2]
        x0 = int(rng.integers(0, w - G + 1)); y0 = int(rng.integers(0, h - G + 1))
        x1 = int(np.clip(x0 + rng.integers(-G // 2, G // 2 + 1), 0, w - G)); y1 = int(rng.integers(0, h - G + 1))
        glob = np.stack([photometric(img[y0:y0 + G, x0:x0 + G], rng), photometric(img[y1:y1 + G, x1:x1 + G], rng)])
        lo, hi = max(0, min(x0, x1) - 200), min(w - L, max(x0, x1) + G + 200 - L)
        loc = np.stack([photometric(img[(yy := int(rng.integers(0, h - L + 1))):yy + L,
                                        (xx := int(rng.integers(lo, hi + 1))):xx + L], rng) for _ in range(NL)])
        return torch.from_numpy(glob), torch.from_numpy(loc)


def block_mask(rng, grid=G // 16, p=0.5, lo=0.1, hi=0.5):
    """iBOT block masking: half the global crops masked, ratio U(0.1, 0.5)."""
    m = np.zeros((grid, grid), bool)
    if rng.random() < p:
        target = int(rng.uniform(lo, hi) * grid * grid)
        for _ in range(100):
            if m.sum() >= target:
                break
            a = rng.integers(4, max(5, target - m.sum() + 1))
            ar = np.exp(rng.uniform(np.log(0.3), np.log(3.3)))
            hh = int(min(grid, max(1, round(np.sqrt(a * ar))))); ww = int(min(grid, max(1, round(np.sqrt(a / ar)))))
            r, c = rng.integers(0, grid - hh + 1), rng.integers(0, grid - ww + 1)
            m[r:r + hh, c:c + ww] = True
    return m.reshape(-1)


# ---------------------------------------------------------------- model parts
class ProjHead(nn.Module):
    """DINO/iBOT projection head; cosine logits against L2-normalised prototypes."""
    def __init__(self, dim, k, hidden=2048, bottleneck=256, bn=True):
        super().__init__()
        # BatchNorm removes the component every image shares (~half the feature energy here), which a
        # random MLP otherwise passes straight through, leaving targets near-uniform (step-0 diagnosis)
        norm = (lambda: nn.BatchNorm1d(hidden)) if bn else (lambda: nn.Identity())
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), norm(), nn.GELU(), nn.Linear(hidden, hidden), norm(), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        self.proto = nn.Parameter(torch.randn(k, bottleneck) * 0.02)

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1) @ F.normalize(self.proto, dim=-1).t()


@torch.no_grad()
def sinkhorn(logits, temp, iters=3):
    q = torch.exp((logits.float() - logits.float().max()) / temp).t()
    k, b = q.shape
    q /= q.sum()
    for _ in range(iters):
        q /= q.sum(1, keepdim=True); q /= k
        q /= q.sum(0, keepdim=True); q /= b
    return (q * b).t()


def koleo(x, eps=1e-8):
    x = F.normalize(x.float(), dim=-1)
    d = torch.cdist(x, x) + torch.eye(len(x), device=x.device) * 1e3
    return -torch.log(d.min(-1).values + eps).mean()


def taps_ln(hidden, ns):
    return [F.layer_norm(hidden[t + 1][:, ns:].float(), (hidden[t + 1].shape[-1],)) for t in TAPS]


def param_groups(model, heads, lr, head_lr, llrd, wd):
    n = model.config.num_hidden_layers
    groups = {}
    for name, p in model.named_parameters():
        if name.startswith("embeddings."):
            depth = 0
        elif name.startswith("model.layer.") or name.startswith("layer."):
            depth = int(name.split("layer.")[1].split(".")[0]) + 1
        else:
            depth = n + 1
        nd = p.ndim <= 1 or "token" in name or "layer_scale" in name
        groups.setdefault((depth, nd), []).append(p)
    out = [{"params": ps, "lr": lr * llrd ** (n + 1 - d), "base_lr": lr * llrd ** (n + 1 - d),
            "weight_decay": 0.0 if nd else wd} for (d, nd), ps in sorted(groups.items())]
    for h in heads:
        out.append({"params": list(h.parameters()), "lr": head_lr, "base_lr": head_lr, "weight_decay": wd})
    return out


# ---------------------------------------------------------------- guards
def load_full(paths, idx):
    return torch.from_numpy(np.stack([((cv2.imread(paths[i])[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD)
                                      .transpose(2, 0, 1) for i in idx]))


@torch.no_grad()
def tap_features(model, x, ns):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model(pixel_values=x, output_hidden_states=True).hidden_states
    return taps_ln(h, ns)


@torch.no_grad()
def drift(model, stock, hold_paths, ns, n=200, bs=4):
    acc = np.zeros(len(TAPS)); k = 0
    for i0 in range(0, n, bs):
        x = load_full(hold_paths, range(i0, min(i0 + bs, n))).cuda()
        a, b = tap_features(model, x, ns), tap_features(stock, x, ns)
        acc += np.array([F.cosine_similarity(u, v, dim=-1).mean().item() for u, v in zip(a, b)]); k += 1
    return (acc / k).round(4).tolist()


def probe(model, tr_paths, tr_ids, hold_paths, hold_ids, ns, n_tr=600, n_ev=200, bs=4, seed=0):
    """Linear patch probe (stride 16, soft targets) on the four tapped layers."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tr_paths), n_tr, replace=False)
    X, Y = [], []
    for i0 in range(0, n_tr, bs):
        idx = pick[i0:i0 + bs]
        f = torch.cat(tap_features(model, load_full(tr_paths, idx).cuda(), ns), -1).half().cpu()
        for k, i in enumerate(idx):
            m = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(tr_ids[i])[0] + ".png"), cv2.IMREAD_UNCHANGED)
            t = soft_targets_at_stride(m, 16).reshape(5, -1).T
            bg = np.flatnonzero(t[:, 0] == 1)
            keep = np.concatenate([np.flatnonzero(t[:, 0] < 1), rng.choice(bg, min(150, len(bg)), replace=False)])
            X.append(f[k, keep]); Y.append(torch.from_numpy(t[keep]))
    X, Y = torch.cat(X).cuda(), torch.cat(Y).float().cuda()           # X stays fp16 on the GPU
    lin = nn.Linear(X.shape[1], 5).cuda()
    opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
    for s in range(400):
        j = torch.randint(0, len(X), (8192,), device="cuda")
        loss = F.cross_entropy(lin(X[j].float()), Y[j]); opt.zero_grad(); loss.backward(); opt.step()
    inter, ps, gs = np.zeros(5), np.zeros(5), np.zeros(5)
    for i0 in range(0, n_ev, bs):
        idx = range(i0, min(i0 + bs, n_ev))
        with torch.no_grad():
            pred = lin(torch.cat(tap_features(model, load_full(hold_paths, idx).cuda(), ns), -1)).argmax(-1).cpu().numpy()
        for k, i in enumerate(idx):
            m = cv2.imread(os.path.join(MASK_DIR, os.path.splitext(hold_ids[i])[0] + ".png"), cv2.IMREAD_UNCHANGED)
            g = soft_targets_at_stride(m, 16).reshape(5, -1).argmax(0)
            for c in range(1, 5):
                inter[c] += ((pred[k] == c) & (g == c)).sum(); ps[c] += (pred[k] == c).sum(); gs[c] += (g == c).sum()
    d = [2 * inter[c] / max(ps[c] + gs[c], 1) for c in range(1, 5)]
    return round(float(np.mean(d)), 4), [round(float(v), 4) for v in d]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["zero", "smoke", "time", "train"])
    ap.add_argument("--model-id", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--head-epochs", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--llrd", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=0.04)
    ap.add_argument("--k-dino", type=int, default=4096)
    ap.add_argument("--k-ibot", type=int, default=4096)
    ap.add_argument("--head-bn", type=int, default=1)
    ap.add_argument("--t-temp", type=float, default=0.04)
    ap.add_argument("--t-temp-final", type=float, default=0.04,
                    help="DINO ramps 0.04->0.07 over ~30 epochs; ramping within a short warm-up flattened every target")
    ap.add_argument("--center", default="softmax", choices=["softmax", "sinkhorn"],
                    help="softmax = DINO v1 EMA centering (works at small batch). sinkhorn forces equal prototype use per "
                         "batch, so with K >> batch every target is near-uniform (smoke test: losses stuck at log K)")
    ap.add_argument("--gram-w", type=float, default=1.0)
    ap.add_argument("--koleo-w", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.994)
    ap.add_argument("--probe-tol", type=float, default=0.01)
    ap.add_argument("--min-cos", type=float, default=0.80)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time-steps", type=int, default=12)
    ap.add_argument("--tag", default="vitb")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-images", type=int, default=200, help="smoke mode only")
    a = ap.parse_args()
    torch.manual_seed(0)
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    stock = AutoModel.from_pretrained(a.model_id).to(dev).eval()
    for q in stock.parameters():
        q.requires_grad_(False)
    ns = 1 + getattr(stock.config, "num_register_tokens", 0)
    dim = stock.config.hidden_size
    hold_ids = pd.read_csv(os.path.join(SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()
    hold_paths = [os.path.join(DATA_DIR, "train_images", f) for f in hold_ids]

    if a.mode == "zero":
        teacher = copy.deepcopy(stock)
        path = os.path.join(OUT, "zero_test.pt")
        torch.save({"backbone": {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}, "epoch": 0}, path)
        fresh = AutoModel.from_pretrained(a.model_id).to(dev).eval()
        fresh.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["backbone"])  # train_full_ft's path
        x = load_full(hold_paths, range(2)).to(dev)
        diffs = [float((u - v).abs().max()) for u, v in zip(tap_features(fresh, x, ns), tap_features(stock, x, ns))]
        os.remove(path)
        print(f"ZERO TEST: max |reloaded - stock| per tap {diffs} -> {'PASS' if max(diffs) == 0 else 'FAIL'}")
        return

    tr_ids = pd.read_csv(os.path.join(SPLIT_DIR, "train12k_ids.csv"))["ImageId"].tolist()
    tr_paths = [os.path.join(DATA_DIR, "train_images", f) for f in tr_ids]
    te_paths = [os.path.join(DATA_DIR, "test_images", f) for f in sorted(os.listdir(os.path.join(DATA_DIR, "test_images")))]
    assert not set(hold_ids) & set(tr_ids), "holdout leaked into the pretraining set"
    paths = tr_paths + te_paths
    if a.mode == "smoke":
        paths = [paths[i] for i in np.random.default_rng(0).choice(len(paths), a.max_images, replace=False)]
    print(f"{len(paths)} unlabelled images ({len(tr_paths)} train12k + {len(te_paths)} test) | holdout {len(hold_ids)} excluded"
          if a.mode != "smoke" else f"SMOKE: {len(paths)} images")

    student = copy.deepcopy(stock).train()
    for q in student.parameters():
        q.requires_grad_(True)
    student.gradient_checkpointing_enable()
    teacher = copy.deepcopy(stock).eval()
    hd_s, hi_s = ProjHead(dim, a.k_dino, bn=bool(a.head_bn)).to(dev), ProjHead(dim, a.k_ibot, bn=bool(a.head_bn)).to(dev)
    hd_t, hi_t = copy.deepcopy(hd_s).train(), copy.deepcopy(hi_s).train()   # BN on batch statistics, as the student
    for m in (teacher, hd_t, hi_t):
        for q in m.parameters():
            q.requires_grad_(False)
    opt = torch.optim.AdamW(param_groups(student, (hd_s, hi_s), a.lr, a.head_lr, a.llrd, a.wd))
    center_d = torch.zeros(1, a.k_dino, device=dev); center_i = torch.zeros(1, a.k_ibot, device=dev)

    loader = torch.utils.data.DataLoader(Views(paths), batch_size=a.bs, shuffle=True, drop_last=True,
                                         num_workers=a.workers, persistent_workers=a.workers > 0)
    spe = len(loader)
    head_steps = max(1, int(round(a.head_epochs * spe))) if a.mode != "time" else 0
    cpt_steps = a.epochs * spe if a.mode == "train" else (spe if a.mode == "smoke" else a.time_steps)
    total = head_steps + cpt_steps
    rng = np.random.default_rng(0)
    start = 0
    if a.resume:
        ck = torch.load(os.path.join(CKPT_DIR, a.resume), map_location=dev, weights_only=False)
        student.load_state_dict(ck["student"]); teacher.load_state_dict(ck["backbone"])
        for mdl, key in ((hd_s, "hd_s"), (hi_s, "hi_s"), (hd_t, "hd_t"), (hi_t, "hi_t")):
            mdl.load_state_dict(ck[key])
        opt.load_state_dict(ck["optimizer"]); start = ck["step"]
        center_d.copy_(ck["center_d"].to(dev)); center_i.copy_(ck["center_i"].to(dev))
        print(f"RESUMED from {a.resume} at step {start}")

    log_path = os.path.join(OUT, f"log_{a.tag}{'_smoke' if a.mode == 'smoke' else ''}.json")
    log = {"args": vars(a), "steps_per_epoch": spe, "head_steps": head_steps, "epochs": []}
    if a.resume and os.path.exists(log_path):
        log = json.load(open(log_path))                      # keeps the stock-probe reference
    if a.mode in ("train", "smoke") and "stock_probe" not in log:
        n_tr, n_ev = (600, 200) if a.mode == "train" else (40, 20)
        ref = probe(stock, tr_paths, tr_ids, hold_paths, hold_ids, ns, n_tr, n_ev)
        log["stock_probe"] = ref
        print(f"stock probe (linear, 4 taps, {n_tr} train / {n_ev} holdout): mean Dice {ref[0]} per class {ref[1]}", flush=True)

    def set_lr(step):
        if step < head_steps:                      # heads only; backbone frozen at lr 0
            for g in opt.param_groups:
                g["lr"] = 0.0 if g is not opt.param_groups[-1] and g is not opt.param_groups[-2] else g["base_lr"]
            return
        s = step - head_steps; warm = max(1, int(0.1 * cpt_steps))
        f = s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, cpt_steps - warm)))
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * f

    step, t0, it = start, time.time(), iter(loader)
    agg = {"dino": 0.0, "ibot": 0.0, "koleo": 0.0, "gram": 0.0, "ent_sample": 0.0, "ent_batch": 0.0}; nb = 0
    while step < total:
        try:
            glob, loc = next(it)
        except StopIteration:
            it = iter(loader); glob, loc = next(it)
        frozen = step < head_steps
        for q in student.parameters():
            q.requires_grad_(not frozen)
        set_lr(step)
        B = glob.shape[0]
        gx = glob.reshape(B * NG, 3, G, G).to(dev, non_blocking=True)
        lx = loc.reshape(B * NL, 3, L, L).to(dev, non_blocking=True)
        mask = torch.from_numpy(np.stack([block_mask(rng) for _ in range(B * NG)])).to(dev)
        t_temp = a.t_temp + (a.t_temp_final - a.t_temp) * step / max(1, total)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                th = teacher(pixel_values=gx, output_hidden_states=True).hidden_states
                td = hd_t(th[-1][:, 0]).float()
                ti = hi_t(th[-1][:, ns:][mask]).float() if int(mask.sum()) > 1 else None
                if a.center == "sinkhorn":
                    t_cls = sinkhorn(td, t_temp); t_pat = sinkhorn(ti, t_temp) if ti is not None else None
                else:
                    t_cls = F.softmax((td - center_d) / t_temp, -1)
                    t_pat = F.softmax((ti - center_i) / t_temp, -1) if ti is not None else None
                a_taps = taps_ln(stock(pixel_values=gx, output_hidden_states=True).hidden_states, ns) if not frozen else None
            sh = student(pixel_values=gx, bool_masked_pos=mask, output_hidden_states=True).hidden_states
            sl = student(pixel_values=lx).last_hidden_state
            s_g, s_l = hd_s(sh[-1][:, 0]).float() / 0.1, hd_s(sl[:, 0]).float() / 0.1
            tc = t_cls.reshape(B, NG, -1)
            sg = F.log_softmax(s_g, -1).reshape(B, NG, -1); slg = F.log_softmax(s_l, -1).reshape(B, NL, -1)
            dino = (-(tc[:, 0] * sg[:, 1]).sum(-1) - (tc[:, 1] * sg[:, 0]).sum(-1)).mean() / 2
            dino = (dino * 2 + sum((-(tc[:, t] * slg[:, v]).sum(-1)).mean() for t in range(NG) for v in range(NL)) / NG) / (2 + NL)
            ibot = (-(t_pat * F.log_softmax(hi_s(sh[-1][:, ns:][mask]).float() / 0.1, -1)).sum(-1)).mean() if t_pat is not None else sh[-1].new_zeros(())
            ko = koleo(sh[-1][:, 0])
            gram = sh[-1].new_zeros(())
            if not frozen:
                # Gram on UNMASKED patches only (masked student tokens are mask-token
                # reconstructions the anchor never saw): zero masked rows/cols in both
                keep = (~mask).float()[..., None]
                n2 = keep.squeeze(-1).sum(1).clamp(min=1) ** 2
                for st, at in zip(taps_ln(sh, ns), a_taps):
                    u, v = F.normalize(st, dim=-1) * keep, F.normalize(at, dim=-1) * keep
                    gram = gram + (((u @ u.transpose(1, 2) - v @ v.transpose(1, 2)) ** 2).sum((1, 2)) / n2).mean()
                gram = gram / len(TAPS)
            loss = dino + ibot + a.koleo_w * ko + a.gram_w * gram
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 3.0)
        opt.step()
        with torch.no_grad():
            m = 1 - (1 - a.ema) * (math.cos(math.pi * step / max(1, total)) + 1) / 2
            pairs = [(hd_t, hd_s), (hi_t, hi_s)] + ([] if frozen else [(teacher, student)])
            center_d.mul_(0.9).add_(td.mean(0, keepdim=True), alpha=0.1)
            if ti is not None:
                center_i.mul_(0.9).add_(ti.mean(0, keepdim=True), alpha=0.1)
            for tm, sm in pairs:
                for pt, ps in zip(tm.parameters(), sm.parameters()):
                    pt.mul_(m).add_(ps.detach(), alpha=1 - m)
            pr = t_cls.float()
            agg["ent_sample"] += float(-(pr * (pr + 1e-12).log()).sum(-1).mean())
            pm = pr.mean(0); agg["ent_batch"] += float(-(pm * (pm + 1e-12).log()).sum())
        for k_, v_ in (("dino", dino), ("ibot", ibot), ("koleo", ko), ("gram", gram)):
            agg[k_] += float(v_.detach())
        nb += 1; step += 1
        if a.mode == "time" and step == 2:
            torch.cuda.synchronize(); t_time = time.time()
        if step % 50 == 0 or step == total or a.mode != "train" and step % 5 == 0:
            el = time.time() - t0
            print(f"step {step}/{total} {'[heads]' if frozen else '[cpt]  '} " + " ".join(f"{k_} {v_ / nb:.4f}" for k_, v_ in agg.items())
                  + f" | log K {math.log(a.k_dino):.2f} | {el / (step - start):.2f}s/step | peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                  flush=True)
            agg = {k_: 0.0 for k_ in agg}; nb = 0
        end_of_epoch = step > head_steps and (step - head_steps) % spe == 0
        if a.mode in ("train", "smoke") and (end_of_epoch or step == head_steps or step == total):
            ep = 0 if step == head_steps and a.mode == "train" else (step - head_steps) // spe
            name = f"cpt_{a.tag}_{'heads' if step == head_steps else f'ep{ep}'}.pt" if a.mode == "train" else None
            if name:
                # slim per-epoch backbone (what --backbone-ckpt loads) + ONE full resume
                # file, overwritten, so six epochs do not cost ~9 GB of disk
                bb = {k_: v_.detach().cpu() for k_, v_ in teacher.state_dict().items()}
                torch.save({"backbone": bb, "epoch": ep, "step": step, "args": vars(a)}, os.path.join(CKPT_DIR, name))
                torch.save({"backbone": bb, "student": {k_: v_.detach().cpu() for k_, v_ in student.state_dict().items()},
                            "hd_s": hd_s.state_dict(), "hi_s": hi_s.state_dict(), "hd_t": hd_t.state_dict(), "hi_t": hi_t.state_dict(),
                            "optimizer": opt.state_dict(), "step": step, "epoch": ep, "args": vars(a),
                            "center_d": center_d.cpu(), "center_i": center_i.cpu()},
                           os.path.join(CKPT_DIR, f"cpt_{a.tag}_resume.pt"))
            if step > head_steps:
                n_tr, n_ev = (600, 200) if a.mode == "train" else (40, 20)
                dc = drift(teacher, stock, hold_paths, ns, n=n_ev)
                pb = probe(teacher, tr_paths, tr_ids, hold_paths, hold_ids, ns, n_tr, n_ev)
                rec = {"epoch": ep, "step": step, "drift_cos": dc, "probe": pb, "checkpoint": name}
                log["epochs"].append(rec)
                print(f"== epoch {ep}: drift cos per tap {dc} | probe mean Dice {pb[0]} (stock {log['stock_probe'][0]}) per class {pb[1]}", flush=True)
                json.dump(log, open(log_path, "w"), indent=1)
                if pb[0] < log["stock_probe"][0] - a.probe_tol or min(dc) < a.min_cos:
                    print(f"== ABORT: probe {pb[0]} vs stock {log['stock_probe'][0]} (tol {a.probe_tol}), min cos {min(dc)} (floor {a.min_cos})", flush=True)
                    return
    if a.mode == "time":
        torch.cuda.synchronize()
        sps = (time.time() - t_time) / (step - 2)
        print(f"TIMING: {sps:.2f} s/step at batch {a.bs} -> {sps * (len(paths) // a.bs) / 60:.1f} min per epoch over "
              f"{len(paths)} images | peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("CPT DONE")


if __name__ == "__main__":
    main()
