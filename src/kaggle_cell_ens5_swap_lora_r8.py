# ==========================================================================
#  Severstal - ensemble kernel v2.  Paste as ONE cell.  Writes submission.csv
#
#  Same decision rule as v1 (kaggle_ensemble_cell.py): members x TTA views averaged as PROBABILITIES, then per
#  class p_c > PIX[c] at full resolution (bilinear) and the class dropped below MINA[c] pixels.  New in v2:
#    * backbones per checkpoint: "backbone_kind" = dinov3 (v1 path, identical numerics) | cradio | qwenvit
#    * both T4s: every member is replicated on each GPU and the test images are split between them in two threads
#    * per-member input normalisation (DINOv3 ImageNet, C-RADIO raw [0,1], Qwen mean/std 0.5)
#  KAGGLE_INPUT / KAGGLE_WORK / LIMIT_IMAGES env vars exist only for local testing.
# ==========================================================================
import os, glob, shutil, time, threading, numpy as np, pandas as pd, cv2, torch, torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

CKPTS     = ['model_vitb.pt', 'model_vits_lora_r16.pt', 'model_bce16.pt', 'model_vitb_lora_warm_pseudo01.pt', 'model_cradio_so_soft_pseudo01.pt']   # V79_swap_lora_r8_for_cradio
WEIGHTS   = None
TTA       = ("none", "h", "v")
PIX       = [0.7, 0.66, 0.65, 0.64]   # rate-matched to V79's test firing rates
MINA      = [630, 1470, 1050, 3150]
BILINEAR  = True
HALF      = True
BS        = 4
GRID, UPS, CLASSES = (16, 100), 4, (1, 2, 3, 4)
INPUT     = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
WORK      = os.environ.get("KAGGLE_WORK", "/kaggle/working")
LIMIT     = int(os.environ.get("LIMIT_IMAGES", "0"))       # 0 = all test images


class HeadNet(nn.Module):
    def __init__(self, dim, hidden, n_classes=5, upscale=4):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(hidden, n_classes * upscale * upscale, 1)
        self.shuffle = nn.PixelShuffle(upscale)

    def forward(self, x):
        return self.shuffle(self.out(self.trunk(x)))


class ProgressiveHead(nn.Module):
    def __init__(self, dim, hidden, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        h2 = hidden // 2
        self.to8 = nn.Conv2d(hidden, h2 * 4, 1)
        self.at8 = nn.Sequential(nn.Conv2d(h2, h2, 3, padding=1), nn.GELU())
        h4 = h2 // 2
        self.to4 = nn.Conv2d(h2, h4 * 4, 1)
        self.at4 = nn.Sequential(nn.Conv2d(h4, h4, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(h4, n_classes, 1)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.stem(x)
        x = self.at8(self.up(self.to8(x)))
        x = self.at4(self.up(self.to4(x)))
        return self.out(x)


class SimplexHead(nn.Module):
    def __init__(self, dim, hidden, n_classes=5, upscale=4, etf_dim=16):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.out = nn.Conv2d(hidden, etf_dim * upscale * upscale, 1)
        self.shuffle = nn.PixelShuffle(upscale)
        self.register_buffer("etf", torch.zeros(etf_dim, n_classes))
        self.log_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        z = F.normalize(self.shuffle(self.out(self.trunk(x))).float(), dim=1)
        return self.log_scale.float().exp().clamp(1.0, 50.0) * torch.einsum("bdhw,dk->bkhw", z, self.etf.float())


def rle_encode(mask):
    p = np.concatenate([[0], mask.flatten(order="F"), [0]])
    r = np.where(p[1:] != p[:-1])[0] + 1
    r[1::2] -= r[::2]
    return " ".join(str(x) for x in r)


assert torch.cuda.is_available(), "Settings > Accelerator > GPU T4 x2"
DEVICES = (os.environ["DEVICES_OVERRIDE"].split(",") if os.environ.get("DEVICES_OVERRIDE")   # local test only
           else [f"cuda:{i}" for i in range(torch.cuda.device_count())])
t0 = time.time()


def find_one(pattern, root=INPUT):
    hits = glob.glob(f"{root}/**/{pattern}", recursive=True)
    assert hits, f"NOT FOUND: {pattern} - dataset not attached, or wrong version"
    return hits[0]


CKPT_PATHS = [find_one(n) for n in CKPTS]
SUB_CSV = find_one("sample_submission.csv")
IMG_DIR = find_one("test_images")
DEFAULT_MID = "facebook/dinov3-vits16-pretrain-lvd1689m"
IMAGENET = (np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32))


def cfg_dir_for(model_id, ckpt_path):
    """v1's DINOv3 config resolution, unchanged (config_<short>.json, else beside the checkpoint, else ViT-S)."""
    short = model_id.split("dinov3-")[-1].split("-")[0]
    hits = glob.glob(f"{INPUT}/**/config_{short}.json", recursive=True)
    if hits:
        d = f"{WORK}/cfg_{short}"
        os.makedirs(d, exist_ok=True)
        shutil.copy(hits[0], f"{d}/config.json")
        return d
    own = os.path.dirname(ckpt_path)
    if os.path.exists(f"{own}/config.json"):
        return own
    assert short == "vits16", f"{os.path.basename(ckpt_path)} needs config_{short}.json in {INPUT}"
    return os.path.dirname(find_one("config.json"))


def remap(sd, want):
    if set(sd) == want:
        return sd
    stripped = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    if set(stripped) == want:
        return stripped
    prefixed = {("model." + k): v for k, v in sd.items()}
    assert set(prefixed) == want, "checkpoint keys match neither layout"
    return prefixed


def adapter_dir(name, with_code):
    """Rebuild a model directory from the flattened dataset: config_<name>.json (+ C-RADIO's .py files)."""
    d = f"{WORK}/bb_{name}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(find_one(f"config_{name}.json"), f"{d}/config.json")
    if with_code:
        src = os.path.dirname(find_one("radio_model.py"))
        for f in glob.glob(f"{src}/*.py"):
            shutil.copy(f, d)
    return d


class Member:
    """One checkpoint on one device: feats(x01) -> (B, 4*dim, 16, 100), from RGB in [0, 1]."""

    def __init__(self, ck, path, device):
        self.device, self.kind = device, ck.get("backbone_kind") or "dinov3"
        taps = tuple(ck["tap_layers"])
        if self.kind == "dinov3":
            mid = ck.get("model_id") or DEFAULT_MID
            cfg = AutoConfig.from_pretrained(cfg_dir_for(mid, path))
            self.bb = AutoModel.from_config(cfg)
            self.bb.load_state_dict(remap(ck["backbone"], set(self.bb.state_dict())), strict=True)
            dim, self.ns, self.norm, self.name = cfg.hidden_size, 1 + getattr(cfg, "num_register_tokens", 0), IMAGENET, mid.split("/")[-1]
        elif self.kind == "cradio":
            d = adapter_dir(ck["backbone_name"], with_code=True)
            cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
            self.bb = AutoModel.from_config(cfg, trust_remote_code=True)
            self.bb.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["backbone"].items()}, strict=True)
            dim, self.norm, self.name = int(self.bb.radio_model.model.embed_dim), (np.zeros(3, np.float32), np.ones(3, np.float32)), ck["backbone_name"]
        elif self.kind == "qwenvit":
            from transformers import Qwen3VLVisionModel
            vc = AutoConfig.from_pretrained(adapter_dir(ck["backbone_name"], with_code=False)).vision_config
            self.bb = Qwen3VLVisionModel(vc)
            self.bb.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["backbone"].items()}, strict=True)
            dim, self.norm, self.name = int(vc.hidden_size), (np.full(3, 0.5, np.float32), np.full(3, 0.5, np.float32)), ck["backbone_name"]
            self.ps, self.tps, self.m = int(vc.patch_size), int(vc.temporal_patch_size), int(vc.spatial_merge_size)
            self.cap = {}
            for idx, mg in zip(self.bb.deepstack_visual_indexes, self.bb.deepstack_merger_list):
                mg.register_forward_pre_hook(lambda _m, inp, i=int(idx): self.cap.__setitem__(i, inp[0]))
            self.bb.merger.register_forward_pre_hook(lambda _m, inp: self.cap.__setitem__(len(self.bb.blocks) - 1, inp[0]))
        else:
            raise ValueError(self.kind)
        self.taps = taps
        arch = ck.get("head_arch", "conv")
        self.hd = (SimplexHead(dim * len(taps), ck["hidden"]) if arch == "simplex"
                   else ProgressiveHead(dim * len(taps), ck["hidden"]) if arch == "progressive"
                   else HeadNet(dim * len(taps), ck["hidden"]))
        self.hd.load_state_dict(ck["head"])
        self.dec = "sigmoid" if ck.get("loss") == "bce_dice" else "softmax"
        self.bb, self.hd = self.bb.to(device).eval(), self.hd.to(device).eval()
        self.mean = torch.tensor(self.norm[0], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.norm[1], device=device).view(1, 3, 1, 1)
        self.info = f"{self.kind}:{self.name} taps {taps} head {arch} decision {self.dec} val {ck.get('val_mean_dice', float('nan')):.4f}"

    def feats(self, x01):
        x = (x01 - self.mean) / self.std
        if self.kind == "dinov3":
            o = self.bb(pixel_values=x, output_hidden_states=True)
            tok = torch.cat([F.layer_norm(o.hidden_states[t + 1][:, self.ns:, :], (o.hidden_states[t + 1].shape[-1],))
                             for t in self.taps], -1)
            return tok.permute(0, 2, 1).reshape(len(x), -1, *GRID)
        if self.kind == "cradio":
            outs = self.bb.radio_model.forward_intermediates(x, indices=list(self.taps), output_fmt="NLC", intermediates_only=True)
            tok = torch.cat([F.layer_norm(o, (o.shape[-1],)) for o in outs], -1)
            return tok.permute(0, 2, 1).reshape(len(x), -1, *GRID)
        B, C, H, W = x.shape                                   # qwenvit: Qwen's patchify, then undo merge-window order
        ps, tps, m = self.ps, self.tps, self.m
        gh, gw = H // ps, W // ps
        t = x.unsqueeze(1).expand(B, tps, C, H, W).reshape(B, 1, tps, C, gh // m, m, ps, gw // m, m, ps)
        pv = t.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9).reshape(B * gh * gw, C * tps * ps * ps)
        self.cap.clear()
        self.bb(pv, grid_thw=torch.tensor([[1, gh, gw]] * B, device=x.device))
        out = []
        for i in self.taps:
            h = self.cap[i].reshape(B, gh // m, gw // m, m, m, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, gh * gw, -1)
            out.append(F.layer_norm(h, (h.shape[-1],)))
        return torch.cat(out, -1).permute(0, 2, 1).reshape(B, -1, gh, gw)


replicas = [[] for _ in DEVICES]          # one member list PER WORKER (not per device name: names may repeat)
for path in CKPT_PATHS:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for w_i, d in enumerate(DEVICES):
        replicas[w_i].append(Member(ck, path, d))
    print(f"loaded {os.path.basename(path)} | {replicas[0][-1].info} | on {len(DEVICES)} worker(s)")
    del ck

sub = pd.read_csv(SUB_CSV)
columns = list(sub.columns)
wide = "ImageId_ClassId" in columns
key = "ImageId_ClassId" if wide else "ImageId"
image_ids = sorted({s.rsplit("_", 1)[0] if wide else s for s in sub[key].astype(str)})
if LIMIT:
    image_ids = image_ids[:LIMIT]
print(f"columns {columns} | {len(image_ids)} images -> {len(image_ids)*4} rows | devices {DEVICES}")

W = [1.0] * len(CKPT_PATHS) if WEIGHTS is None else [float(w) for w in WEIGHTS]
assert len(W) == len(CKPT_PATHS) and all(len(r) == len(W) for r in replicas), "members and weights out of step"


@torch.no_grad()
def probs(x01, members):
    acc, tot = 0.0, 0.0
    for mb, w in zip(members, W):
        for v in TTA:
            xv = x01 if v == "none" else torch.flip(x01, [3] if v == "h" else [2])
            lg = mb.hd(mb.feats(xv)).float()
            if mb.dec == "sigmoid":
                p = torch.sigmoid(lg)
                p[:, 0] = 0.0
            else:
                p = lg.softmax(1)
            if v != "none":
                p = torch.flip(p, [3] if v == "h" else [2])
            acc = acc + w * p
            tot += w
    return acc / tot


results, kept_total, done = {}, [0], [0]
lock = threading.Lock()


def worker(w_i, dev, ids):
    members = replicas[w_i]
    torch.cuda.set_device(dev)
    for i0 in range(0, len(ids), BS):
        ch = ids[i0:i0 + BS]
        x = np.stack([(cv2.imread(f"{IMG_DIR}/{f}")[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1) for f in ch])
        x = torch.from_numpy(x).to(dev)
        with (torch.autocast("cuda", dtype=torch.float16) if HALF else torch.autocast("cuda", enabled=False)):
            p = probs(x, members)
        p = p.cpu().numpy()
        for k, image_id in enumerate(ch):
            out = []
            for j, c in enumerate(CLASSES):
                if BILINEAR:
                    full = cv2.resize(p[k, c], (1600, 256), interpolation=cv2.INTER_LINEAR) > PIX[j]
                    area = full.sum()
                else:
                    small = p[k, c] > PIX[j]
                    area = small.sum() * (UPS * UPS)
                if area < MINA[j]:
                    rle = ""
                else:
                    full = full if BILINEAR else np.repeat(np.repeat(small, UPS, 0), UPS, 1)
                    rle = rle_encode(full.astype(np.uint8))
                out.append((c, rle))
            with lock:
                results[image_id] = out
                kept_total[0] += sum(1 for _, r in out if r)
        with lock:
            done[0] += len(ch)
            n = done[0]
        if n % 500 < BS:
            el = time.time() - t0
            print(f"  {n}/{len(image_ids)}  {el:.0f}s, ~{el / n * len(image_ids):.0f}s projected", flush=True)


chunks = [image_ids[i::len(DEVICES)] for i in range(len(DEVICES))]
threads = [threading.Thread(target=worker, args=(i, d, ids)) for i, (d, ids) in enumerate(zip(DEVICES, chunks))]
for th in threads:
    th.start()
for th in threads:
    th.join()
assert len(results) == len(image_ids), f"only {len(results)} of {len(image_ids)} images processed"

rows = []
for image_id in image_ids:                                    # same row order as v1
    for c, rle in results[image_id]:
        rows.append({"ImageId_ClassId": f"{image_id}_{c}", "EncodedPixels": rle} if wide
                    else {"ImageId": image_id, "EncodedPixels": rle, "ClassId": c})
df = pd.DataFrame(rows)[columns]
df.to_csv(os.path.join(WORK, "submission.csv") if WORK != "/kaggle/working" else "submission.csv", index=False)
n_pairs = len(image_ids) * 4
print(f"\nwrote submission.csv: {len(df)} rows, {kept_total[0]} non-empty = {100*kept_total[0]/n_pairs:.2f}% predicted-positive rate")
print(f"total {time.time()-t0:.0f}s on {len(DEVICES)} GPU(s)")
