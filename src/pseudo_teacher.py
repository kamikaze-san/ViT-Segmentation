"""Pseudo-label teacher: the exact 0.90756 submission ensemble run over the 5,506
competition test images. Trains nothing - build_pseudo_labels.py turns these
arrays into masks.

Writes to outputs/pseudo/:
  probs.npy       (n, 4, 64, 400) float16  member- and TTA-averaged P(class c),
                                           classes 1..4, stride 4 (what the
                                           submission thresholds after bilinear)
  ens_dec.npy     (n, 4) bool              the submission's own decisions
  member_dec.npy  (5, n, 4) bool           each member's OWN TTA-averaged
                                           decision - unanimity needs the members
                                           separately, and the average destroys that
  member_max.npy  (5, n, 4) float16        each member's max probability
  progress.json   resume point + the config it was produced with

Resumable: arrays are flushed and progress recorded every --flush-every batches;
a restart continues from the last record and refuses to mix configs.

--verify N  correctness check before any real run: loads the five bundle files,
runs them single-view fp32 at E17's thresholds on N test images where the
members DISAGREE, and compares against outputs/complement/dec_*.npy. Images
where every member says empty would pass even with a broken model, so they are
skipped on purpose.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from spatial_head import build
from train_lora import DATA_DIR, GRID

OUTPUTS = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs"
OUT = os.path.join(OUTPUTS, "pseudo")
IMG_DIR = os.path.join(DATA_DIR, "test_images")

MEMBERS = [   # (bundle file, E17 decision-cache name) - the 0.90756 submission, same order
    ("kaggle_bundle_vitb/model_vitb.pt",           "vitb"),
    ("kaggle_bundle_lora/model_vits_lora_r16.pt",  "vits_lora_r16"),
    ("kaggle_bundle_lora/model_lora_r8.pt",        "lora_r8"),
    ("kaggle_bundle_bce/model_bce16.pt",           "bce16"),
    ("kaggle_bundle_vitb/model_vitb_lora_warm.pt", "vitb_lora_warm"),
]
TTA = ("none", "h", "v")
PIX, MINA = [0.65, 0.60, 0.65, 0.58], [630, 1470, 1050, 3150]            # 0.90756
E17_PIX, E17_MINA = [0.60, 0.60, 0.60, 0.45], [1050, 2450, 1750, 5250]   # dec_*.npy
DEFAULT_MID = "facebook/dinov3-vits16-pretrain-lvd1689m"   # pre-dates the model_id key
MEAN = np.array([0.485, 0.456, 0.406], np.float32)         # as the Kaggle cell
STD = np.array([0.229, 0.224, 0.225], np.float32)


def remap(sd, want):
    """Same key-layout tolerance as the Kaggle cell (model.layer.N vs layer.N)."""
    if set(sd) == want:
        return sd
    stripped = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    if set(stripped) == want:
        return stripped
    prefixed = {("model." + k): v for k, v in sd.items()}
    assert set(prefixed) == want, "checkpoint keys match neither layout"
    return prefixed


def load_member(rel, device):
    ck = torch.load(os.path.join(OUTPUTS, rel), map_location="cpu", weights_only=False)
    mid = ck.get("model_id") or DEFAULT_MID
    cfg = AutoConfig.from_pretrained(mid)
    bb = AutoModel.from_config(cfg)
    bb.load_state_dict(remap(ck["backbone"], set(bb.state_dict())), strict=True)
    taps = tuple(ck["tap_layers"])
    hd = build(ck.get("head_arch", "conv"), dim=cfg.hidden_size * len(taps),
               upscale=4, hidden=ck["hidden"])
    hd.load_state_dict(ck["head"])
    sig = ck.get("loss") == "bce_dice"   # sigmoid model scored by softmax = silent wrong probs
    ns = 1 + getattr(cfg, "num_register_tokens", 0)
    print(f"  {os.path.basename(rel):<26} {mid.split('/')[-1]:<32} taps {taps} "
          f"head {ck.get('head_arch', 'conv')} {'sigmoid' if sig else 'softmax'}", flush=True)
    return bb.to(device).eval(), hd.to(device).eval(), taps, ns, sig


def load_images(files):
    return torch.from_numpy(np.stack([
        (((cv2.imread(os.path.join(IMG_DIR, f))[:, :, ::-1].astype(np.float32) / 255.0)
          - MEAN) / STD).transpose(2, 0, 1) for f in files]))


@torch.no_grad()
def member_probs(net, x, views, half):
    """One member, averaged over its TTA views. (B, 5, 64, 400) float32 numpy."""
    bb, hd, taps, ns, sig = net
    acc = 0.0
    for v in views:
        xv = x if v == "none" else torch.flip(x, [3] if v == "h" else [2])
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=half):
            o = bb(pixel_values=xv, output_hidden_states=True)
            tok = torch.cat([F.layer_norm(o.hidden_states[t + 1][:, ns:, :],
                                          (o.hidden_states[t + 1].shape[-1],)) for t in taps], -1)
            lg = hd(tok.permute(0, 2, 1).reshape(len(x), -1, *GRID)).float()
        if sig:
            p = torch.sigmoid(lg)
            p[:, 0] = 0.0                     # no background class under sigmoid
        else:
            p = lg.softmax(1)
        if v != "none":
            p = torch.flip(p, [3] if v == "h" else [2])
        acc = acc + p
    return (acc / len(views)).cpu().numpy()


def decide(p, pix, mina):
    """Submission rule: bilinear to 256x1600, > pix, area >= min_area."""
    d = np.zeros((len(p), 4), bool); mx = np.zeros((len(p), 4), np.float16)
    for k in range(len(p)):
        for j in range(4):
            full = cv2.resize(p[k, j + 1], (1600, 256), interpolation=cv2.INTER_LINEAR)
            d[k, j] = (full > pix[j]).sum() >= mina[j]
            mx[k, j] = p[k, j + 1].max()
    return d, mx


def verify(files, nets, n_check, device, bs):
    refs = np.stack([np.load(os.path.join(OUTPUTS, "complement", f"dec_{name}.npy"))
                     for _, name in MEMBERS])                      # (5, 5506, 4)
    votes = refs.sum(0)                                            # (5506, 4)
    informative = np.where(((votes > 0) & (votes < len(MEMBERS))).any(1))[0][:n_check]
    print(f"verifying on {len(informative)} images where members disagree "
          f"({len(informative) * 4 * len(MEMBERS)} member decisions)")
    ok = True
    for m, ((rel, name), net) in enumerate(zip(MEMBERS, nets)):
        got = []
        for i0 in range(0, len(informative), bs):
            idx = informative[i0:i0 + bs]
            x = load_images([files[i] for i in idx]).to(device)
            got.append(decide(member_probs(net, x, ("none",), half=False), E17_PIX, E17_MINA)[0])
        got = np.concatenate(got); ref = refs[m, informative]
        bad = int((got != ref).sum())
        ok &= bad == 0
        print(f"  {name:<16} {'matches E17' if bad == 0 else f'MISMATCH {bad}/{got.size}'}"
              f"   positives got {int(got.sum())} ref {int(ref.sum())}", flush=True)
    print("VERIFY PASSED" if ok else "VERIFY FAILED - do not run the teacher")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--half", type=int, default=1, help="fp16 autocast, as the Kaggle cell")
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--verify", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    files = sorted(os.listdir(IMG_DIR))          # same order as complementarity.py
    print(f"{len(files)} test images | device {a.device}")
    nets = [load_member(rel, a.device) for rel, _ in MEMBERS]
    if a.verify:
        return verify(files, nets, a.verify, a.device, a.bs)

    files = files[:a.limit] if a.limit else files
    n, M = len(files), len(MEMBERS)
    os.makedirs(OUT, exist_ok=True)
    cfg = {"members": [r for r, _ in MEMBERS], "tta": list(TTA), "pix": PIX, "mina": MINA,
           "half": a.half, "n": n}
    paths = {k: os.path.join(OUT, f"{k}.npy") for k in ("probs", "ens_dec", "member_dec", "member_max")}
    prog = os.path.join(OUT, "progress.json")
    if os.path.exists(prog):
        saved = json.load(open(prog))
        assert {k: saved[k] for k in cfg} == cfg, "outputs/pseudo was made with a different config"
        arr = {k: np.lib.format.open_memmap(p, mode="r+") for k, p in paths.items()}
        start = saved["next"]
        print(f"resuming at image {start}/{n}")
    else:
        shapes = {"probs": ((n, 4, 64, 400), np.float16), "ens_dec": ((n, 4), bool),
                  "member_dec": ((M, n, 4), bool), "member_max": ((M, n, 4), np.float16)}
        arr = {k: np.lib.format.open_memmap(paths[k], mode="w+", dtype=d, shape=s)
               for k, (s, d) in shapes.items()}
        pd.DataFrame({"ImageId": files}).to_csv(os.path.join(OUT, "test_ids.csv"), index=False)
        start = 0

    t0 = time.time()
    for b, i0 in enumerate(range(start, n, a.bs)):
        ch = files[i0:i0 + a.bs]; sl = slice(i0, i0 + len(ch))
        x = load_images(ch).to(a.device)
        tot = 0.0
        for m, net in enumerate(nets):
            p = member_probs(net, x, TTA, half=bool(a.half))
            arr["member_dec"][m, sl], arr["member_max"][m, sl] = decide(p, PIX, MINA)
            tot = tot + p
        avg = tot / M
        arr["probs"][sl] = avg[:, 1:].astype(np.float16)
        arr["ens_dec"][sl] = decide(avg, PIX, MINA)[0]
        if (b + 1) % a.flush_every == 0 or i0 + len(ch) >= n:
            for v in arr.values():
                v.flush()
            json.dump({**cfg, "next": i0 + len(ch)}, open(prog, "w"))   # only after flush
            done = i0 + len(ch) - start; el = time.time() - t0
            print(f"  {i0 + len(ch)}/{n} | {el / 60:5.1f} min | "
                  f"ETA {(n - i0 - len(ch)) * el / done / 60:5.1f} min", flush=True)
    e = np.asarray(arr["ens_dec"]); md = np.asarray(arr["member_dec"])
    print(f"\nsubmission positive rate per class: {(100 * e.mean(0)).round(2)}")
    print(f"images with all classes unanimous across {M} members: "
          f"{int(((md.all(0) | (~md).all(0)).all(1)).sum())}/{n}")


if __name__ == "__main__":
    main()
