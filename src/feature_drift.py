"""E31 follow-up, job 2 (analysis): how far does early stage-2 LoRA move each backbone's features, per learning rate?

Reads the checkpoints of the short runs made by radio_checks.sh (1 epoch at constant peak LR from the stage-1 head,
labelled strips only) and compares their tap features with the stock backbone's on the first K holdout strips, in eval
mode, fp32, exactly as the head reads them (LayerNorm'd per tap):
  rel  ||f_run - f_stock|| / ||f_stock|| per token, averaged   (0 = unchanged)
  cos  cosine(f_run, f_stock) per token, averaged               (1 = unchanged)
plus each run's holdout mean Dice before (epoch 0) and after (epoch 1), parsed from its log.
Questions: at the same LR, does C-RADIO's representation move much more than ViT-B's (the open hypothesis for its
0.69 -> 0.56-0.60 stage-2 drop)? Which C-RADIO LR moves it as much as ViT-B moves at 1e-4?
  python feature_drift.py --arm cradio:1e-4:_lrdrift_cradio_1e-4 [--arm dinov3:1e-4:_lrdrift_vitb_1e-4 ...] [--k 32]
"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd, torch, torch.nn.functional as F

import train_full_ft as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs", "lr_drift")
LOGS = os.path.join(HERE, "..", "outputs", "logs")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_stock(kind, ck, dev):
    """Stock backbone + its input normalisation, with train_full_ft's globals set so T.tap_features reads its taps."""
    if kind == "dinov3":
        from transformers import AutoModel, AutoImageProcessor
        mid = ck.get("model_id") or "facebook/dinov3-vitb16-pretrain-lvd1689m"
        proc = AutoImageProcessor.from_pretrained(mid)
        model = AutoModel.from_pretrained(mid).to(dev).eval()
        T.ADAPTER, T.TAP_LAYERS = None, tuple(ck.get("tap_layers") or (2, 5, 8, 11))
        norm = {"mean": np.array(proc.image_mean, np.float32), "std": np.array(proc.image_std, np.float32)}
        return model, norm, 1 + getattr(model.config, "num_register_tokens", 0)
    from backbones import build_backbone
    T.ADAPTER = build_backbone(kind, ck["backbone_path"], dev)
    T.TAP_LAYERS = tuple(T.ADAPTER.taps)
    norm = {"mean": np.array(T.ADAPTER.norm["mean"], np.float32), "std": np.array(T.ADAPTER.norm["std"], np.float32)}
    return T.ADAPTER.module.eval(), norm, 0


@torch.no_grad()
def tap_maps(model, ids, norm, ns, dev, bs=4):
    out = []
    for i in range(0, len(ids), bs):
        x, _, _ = T.read_batch(ids, np.arange(i, min(i + bs, len(ids))), 4, norm)
        out.append(T.tap_features(model, torch.from_numpy(x).to(dev), ns).float().cpu())
    return torch.cat(out)                              # (K, 4*dim, 16, 100)


def drift(f0, f1, n_taps=4):
    d, res = f0.shape[1] // n_taps, []
    for t in range(n_taps):
        a, b = f0[:, t * d:(t + 1) * d], f1[:, t * d:(t + 1) * d]
        rel = ((b - a).norm(dim=1) / a.norm(dim=1).clamp_min(1e-6)).mean().item()
        res.append({"rel": round(rel, 4), "cos": round(F.cosine_similarity(a, b, dim=1).mean().item(), 4)})
    return res


def parse_log(tag):
    path = os.path.join(LOGS, f"lrdrift{tag}.log")
    txt = open(path, encoding="utf-8", errors="replace").read() if os.path.exists(path) else ""
    e0 = re.search(r"epoch 0 \(head as-loaded.*?val mean ([0-9.]+)", txt)
    e1 = re.search(r"^ep\s+1 \| loss ([0-9.]+) .*?val c1 (\S+) c2 (\S+) c3 (\S+) c4 (\S+) \| mean ([0-9.]+)", txt, re.M)
    return {"ep0_mean": float(e0.group(1)) if e0 else None,
            "ep1_mean": float(e1.group(6)) if e1 else None,
            "ep1_per_class": [float(v) for v in e1.group(2, 3, 4, 5)] if e1 else None,
            "ep1_loss": float(e1.group(1)) if e1 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, help="kind:lr:tag, e.g. cradio:1e-4:_lrdrift_cradio_1e-4")
    ap.add_argument("--k", type=int, default=32, help="holdout strips compared")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = pick_device()
    ids = pd.read_csv(os.path.join(T.SPLIT_DIR, "holdout_ids.csv"))["ImageId"].tolist()[:a.k]
    arms = []
    for s in a.arm:
        kind, lr, tag = s.split(":", 2)
        hits = [p for p in glob.glob(os.path.join(T.CKPT_DIR, f"fullft_lr*{tag}.pt")) if "_ep" not in os.path.basename(p)]
        assert len(hits) == 1, f"arm {s}: expected one checkpoint, found {hits}"
        arms.append({"kind": kind, "lr": float(lr), "tag": tag, "ckpt": hits[0]})

    rows = []
    for kind in dict.fromkeys(x["kind"] for x in arms):
        group = [x for x in arms if x["kind"] == kind]
        ck0 = torch.load(group[0]["ckpt"], map_location="cpu", weights_only=False, mmap=True)
        model, norm, ns = load_stock(kind, ck0, dev)
        f0 = tap_maps(model, ids, norm, ns, dev)
        print(f"{kind}: stock taps {T.TAP_LAYERS}, features {tuple(f0.shape)} on {len(ids)} holdout strips ({dev})", flush=True)
        for x in group:
            ck = torch.load(x["ckpt"], map_location="cpu", weights_only=False, mmap=True)
            # the best-epoch file (save_best) has no lora_r key but its backbone is the merged LoRA weights
            assert ck.get("backbone") is not None, f"{x['ckpt']} has no backbone (frozen run?)"
            model.load_state_dict(ck["backbone"]); model.eval()
            dr = drift(f0, tap_maps(model, ids, norm, ns, dev))
            row = {**x, **parse_log(x["tag"]), "taps": list(T.TAP_LAYERS), "per_tap": dr,
                   "rel_mean": round(float(np.mean([d["rel"] for d in dr])), 4),
                   "cos_mean": round(float(np.mean([d["cos"] for d in dr])), 4)}
            if row["ep0_mean"] is not None and row["ep1_mean"] is not None:
                row["drop"] = round(row["ep1_mean"] - row["ep0_mean"], 4)
            rows.append(row)
            print(f"  lr {x['lr']:.0e}: rel per tap {[d['rel'] for d in dr]} | cos {[d['cos'] for d in dr]} | "
                  f"holdout {row['ep0_mean']} -> {row['ep1_mean']}", flush=True)
            del ck
        del model, f0
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    res = {"k_holdout": len(ids), "device": str(dev), "arms": rows}
    ref = [r for r in rows if r["kind"] == "dinov3" and abs(r["lr"] - 1e-4) < 1e-12]
    cr = sorted([r for r in rows if r["kind"] == "cradio"], key=lambda r: r["lr"])
    if ref and len(cr) >= 2:
        target = ref[0]["rel_mean"]
        lr_, d_ = np.log([r["lr"] for r in cr]), np.log([max(r["rel_mean"], 1e-8) for r in cr])
        if d_[0] <= np.log(target) <= d_[-1]:
            res["cradio_lr_matching_vitb_1e-4_drift"] = float(np.exp(np.interp(np.log(target), d_, lr_)))
        else:
            res["cradio_lr_matching_vitb_1e-4_drift"] = (f"outside the tested range: ViT-B drift {target} vs C-RADIO "
                                                         f"{cr[0]['rel_mean']}-{cr[-1]['rel_mean']}")
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    print(f"\n{'arm':8s} {'lr':>7s} {'rel drift (mean of taps)':>25s} {'cos':>7s} {'holdout ep0 -> ep1':>20s} {'drop':>8s}")
    for r in rows:
        hd = f"{r['ep0_mean']} -> {r['ep1_mean']}"
        print(f"{r['kind']:8s} {r['lr']:7.0e} {r['rel_mean']:25.4f} {r['cos_mean']:7.4f} {hd:>20s} {r.get('drop', float('nan')):+8.4f}")
    if "cradio_lr_matching_vitb_1e-4_drift" in res:
        print(f"C-RADIO LR with ViT-B@1e-4's drift: {res['cradio_lr_matching_vitb_1e-4_drift']}")
    print(os.path.join(OUT, "summary.json"), flush=True)


if __name__ == "__main__":
    main()
