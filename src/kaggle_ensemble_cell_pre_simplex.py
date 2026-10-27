# ==========================================================================
#  Severstal - DINOv3 ensemble kernel.  Paste as ONE cell.  Writes submission.csv
#
#  Two checkpoints x 3 TTA views = 6 forward passes, averaged as PROBABILITIES
#  before thresholding.  Averaging probabilities is what was validated on the
#  500-image holdout (+0.00133); thresholding each model and voting is a
#  different, unvalidated rule.
#
#  Per-class decision (NOT argmax): a pixel joins class c if p_c > PIX[c], and
#  the whole mask is dropped if it covers fewer than MINA[c] pixels.  Classes
#  are independent under the Kaggle metric, so they are decided independently.
# ==========================================================================
import os, glob, shutil, time, numpy as np, pandas as pd, cv2, torch, torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

CKPTS     = ["model_vitb.pt", "model_12k_ep16.pt"]   # different backbone AND different aug
WEIGHTS   = None     # per-member weight, same order as CKPTS. None = uniform (as before).
                     # Members are averaged with EQUAL weight today despite spanning
                     # 0.89961 to 0.90424 in measured target quality, so the best model
                     # gets no more say than the worst. Weights are normalised, so only
                     # their ratios matter: [1,1,1,1,2] doubles the last member.
TTA       = ("none", "h", "v")
PIX       = [0.65, 0.65, 0.65, 0.55]        # midpoint: ViT-B wants .70, ViT-S .60/.45
MINA      = [1050, 2450, 2450, 5250]        # full-res pixels; see rate screen
BILINEAR  = True     # interpolate probs to full res instead of block-replicating.
                     # Worth +0.00031 on target (0.90147 -> 0.90178). Default ON now:
                     # leaving it off silently cost the ViT-B solo run that much.
HALF      = True                             # fp16: 1hr kernel limit, ~35min instead of ~60
BS        = 4
GRID, UPS, CLASSES = (16, 100), 4, (1, 2, 3, 4)


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
    """stride 16 -> 8 -> 4 with a 3x3 conv AFTER each x2 upsample.

    HeadNet reaches stride 4 in ONE PixelShuffle, so every pixel inside a 4x4
    block comes from the same 1x1 projection of the same stride-16 feature with no
    spatial computation afterwards. This computes at stride 8 and stride 4 as well.
    Selected per checkpoint via ck["head_arch"].
    """

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


def rle_encode(mask):
    """Kaggle RLE: 1-indexed (start, length) pairs, column-major."""
    p = np.concatenate([[0], mask.flatten(order="F"), [0]])
    r = np.where(p[1:] != p[:-1])[0] + 1
    r[1::2] -= r[::2]
    return " ".join(str(x) for x in r)


# A silent CPU fallback would not error - it would just run past the 1hr limit
# and return nothing, which is the most expensive way to fail here.
assert torch.cuda.is_available(), "Settings > Accelerator > GPU T4"
device = "cuda"
t0 = time.time()


# Resolve by recursive glob rather than fixed dataset names: mount points depend
# on how each dataset was attached, and a wrong guess fails at minute 40.
def find_one(pattern, root="/kaggle/input"):
    hits = glob.glob(f"{root}/**/{pattern}", recursive=True)
    assert hits, f"NOT FOUND: {pattern} - dataset not attached, or wrong version"
    return hits[0]


CKPT_PATHS = [find_one(n) for n in CKPTS]      # both, before any compute happens
MODEL_DIR = os.path.dirname(CKPT_PATHS[0])
SUB_CSV = find_one("sample_submission.csv")
IMG_DIR = find_one("test_images")

# Config is resolved PER CHECKPOINT from its stored model_id, so ViT-S and ViT-B
# members can coexist. make_kaggle_bundle.py writes each backbone's config into a
# cfg_<short>/ subdirectory for exactly this reason; a bare glob for config.json
# would return whichever it hit first and silently build the wrong architecture.
DEFAULT_MID = "facebook/dinov3-vits16-pretrain-lvd1689m"   # pre-dates the model_id key


def cfg_dir_for(model_id, ckpt_path):
    """Resolve this backbone's config: a config_<short>.json named for it, else a
    config.json beside this checkpoint, else any config.json in the input tree.

    The naming is what keeps this safe with several datasets attached. ViT-B ships
    as config_vitb16.json, so the generic config.json fallback can only ever
    return a ViT-S config - it can never hand a 768-dim config to a 384-dim
    checkpoint."""
    short = model_id.split("dinov3-")[-1].split("-")[0]     # vits16 / vitb16
    # Kaggle flattens directories, so backbones are disambiguated by FILENAME.
    # AutoConfig needs a directory, so copy it into one under /kaggle/working.
    hits = glob.glob(f"/kaggle/input/**/config_{short}.json", recursive=True)
    if hits:
        d = f"/kaggle/working/cfg_{short}"
        os.makedirs(d, exist_ok=True)
        shutil.copy(hits[0], f"{d}/config.json")
        return d
    # Each checkpoint may live in its OWN single-file dataset with no config
    # beside it (that is how this project's datasets are actually laid out), so
    # fall back to any config.json in the input tree. Safe because ViT-B's config
    # is named config_vitb16.json - a glob for "config.json" cannot return it.
    own = os.path.dirname(ckpt_path)
    if os.path.exists(f"{own}/config.json"):
        return own
    # The generic fallback is ONLY safe for ViT-S, because the bare config.json in
    # this project's datasets is the ViT-S one. Falling back for any other backbone
    # silently builds a 12-layer 384-dim model, and the failure then surfaces as an
    # opaque "checkpoint keys match neither layout" from remap() rather than as the
    # missing file it actually is.
    assert short == "vits16", (
        f"{os.path.basename(ckpt_path)} needs config_{short}.json and it is not in "
        f"/kaggle/input. Upload it alongside the checkpoint - do NOT let this fall "
        f"back to the ViT-S config.json.")
    return os.path.dirname(find_one("config.json"))


MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet, as trained
STD  = np.array([0.229, 0.224, 0.225], np.float32)

def remap(sd, want):
    """transformers 5.0.0 names blocks `model.layer.N.*`; older builds use
    `layer.N.*`. Kaggle's version is not ours, so try both orientations and
    verify against the instantiated model rather than assuming either."""
    if set(sd) == want:
        return sd
    stripped = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    if set(stripped) == want:
        return stripped
    prefixed = {("model." + k): v for k, v in sd.items()}
    assert set(prefixed) == want, "checkpoint keys match neither layout"
    return prefixed


nets = []
for path in CKPT_PATHS:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mid = ck.get("model_id") or DEFAULT_MID
    cfg = AutoConfig.from_pretrained(cfg_dir_for(mid, path))
    bb = AutoModel.from_config(cfg)
    bb.load_state_dict(remap(ck["backbone"], set(bb.state_dict())), strict=True)
    taps = tuple(ck["tap_layers"])
    dim = cfg.hidden_size                       # 384 ViT-S, 768 ViT-B
    ns = 1 + getattr(cfg, "num_register_tokens", 0)
    arch = ck.get("head_arch", "conv")
    hd = (ProgressiveHead(dim * len(taps), ck["hidden"]) if arch == "progressive"
          else HeadNet(dim * len(taps), ck["hidden"]))
    hd.load_state_dict(ck["head"])
    # A bce_dice model has no background channel and its classes do not compete:
    # P(class c) = sigmoid(logit_c) and the four values do not sum to 1. Scoring it
    # with softmax would not crash - it would silently produce wrong probabilities,
    # so this is read from the checkpoint rather than left to a flag.
    dec = "sigmoid" if ck.get("loss") == "bce_dice" else "softmax"
    nets.append((bb.to(device).eval(), hd.to(device).eval(), taps, ns, dec))
    print(f"loaded {os.path.basename(path)} | {mid.split('/')[-1]} dim {dim} "
          f"| taps {taps} | head {arch} | decision {dec} "
          f"| val {ck.get('val_mean_dice', float('nan')):.4f}")

sub = pd.read_csv(SUB_CSV)
columns = list(sub.columns)
wide = "ImageId_ClassId" in columns
key = "ImageId_ClassId" if wide else "ImageId"
image_ids = sorted({s.rsplit("_", 1)[0] if wide else s for s in sub[key].astype(str)})
print(f"columns {columns} | {len(image_ids)} images -> {len(image_ids)*4} rows")


W = [1.0] * len(nets) if WEIGHTS is None else [float(w) for w in WEIGHTS]
assert len(W) == len(nets), f"WEIGHTS has {len(W)} entries for {len(nets)} checkpoints"
print(f"member weights (normalised): {[round(w / sum(W), 4) for w in W]}")


@torch.no_grad()
def probs(x):
    """Weighted mean over all models x all TTA views, at stride 4."""
    acc, tot = 0.0, 0.0
    for (bb, hd, taps, ns, dec), w in zip(nets, W):
        for v in TTA:
            xv = x if v == "none" else torch.flip(x, [3] if v == "h" else [2])
            o = bb(pixel_values=xv, output_hidden_states=True)
            tok = torch.cat([F.layer_norm(o.hidden_states[t + 1][:, ns:, :],
                                          (o.hidden_states[t + 1].shape[-1],)) for t in taps], -1)
            lg = hd(tok.permute(0, 2, 1).reshape(len(x), -1, *GRID)).float()
            if dec == "sigmoid":
                p = torch.sigmoid(lg)
                p[:, 0] = 0.0            # no background class under sigmoid
            else:
                p = lg.softmax(1)
            if v != "none":
                p = torch.flip(p, [3] if v == "h" else [2])
            acc = acc + w * p
            tot += w
    return acc / tot


rows, kept = [], 0
for i0 in range(0, len(image_ids), BS):
    ch = image_ids[i0:i0 + BS]
    x = np.stack([(((cv2.imread(f"{IMG_DIR}/{f}")[:, :, ::-1].astype(np.float32) / 255.0)
                    - MEAN) / STD).transpose(2, 0, 1) for f in ch])
    x = torch.from_numpy(x).to(device)
    with (torch.autocast("cuda", dtype=torch.float16) if HALF
          else torch.autocast("cuda", enabled=False)):
        p = probs(x)
    p = p.cpu().numpy()
    for k, image_id in enumerate(ch):
        for j, c in enumerate(CLASSES):
            # probability is constant across each 4x4 block, so full-res area is
            # exactly 16x the cell count - no need to materialise 256x1600 for
            # the ~89% of pairs that are about to be dropped anyway
            if BILINEAR:
                # We predict on a 16x100 grid; block-replicating to 256x1600 makes
                # every boundary a 4px staircase against a smooth true edge.
                # Interpolate the PROBABILITY map, then threshold at full res.
                full = cv2.resize(p[k, c], (1600, 256), interpolation=cv2.INTER_LINEAR) > PIX[j]
                area = full.sum()
            else:
                small = p[k, c] > PIX[j]
                area = small.sum() * (UPS * UPS)
            if area < MINA[j]:
                rle = ""
            else:
                kept += 1
                full = full if BILINEAR else np.repeat(np.repeat(small, UPS, 0), UPS, 1)
                rle = rle_encode(full.astype(np.uint8))
            rows.append({"ImageId_ClassId": f"{image_id}_{c}", "EncodedPixels": rle} if wide
                        else {"ImageId": image_id, "EncodedPixels": rle, "ClassId": c})
    if (i0 + BS) % 500 < BS:
        el = time.time() - t0
        print(f"  {i0+len(ch)}/{len(image_ids)}  {el:.0f}s, ~{el/(i0+len(ch))*len(image_ids):.0f}s projected",
              flush=True)

df = pd.DataFrame(rows)[columns]
df.to_csv("submission.csv", index=False)
n_pairs = len(image_ids) * 4
print(f"\nwrote submission.csv: {len(df)} rows, {kept} non-empty "
      f"= {100*kept/n_pairs:.2f}% predicted-positive rate")
print(f"(submitted single-model operating point was 11.2% - a large drift means "
      f"MINA needs rescaling)")
print(f"total {time.time()-t0:.0f}s")
