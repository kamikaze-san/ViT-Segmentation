"""Build the Kaggle Dataset bundle: slim checkpoint + model config.

The training checkpoints carry AdamW optimizer state (two fp32 moments over
21.6M params, ~250 MB) which is dead weight for inference. Stripping it takes
the upload from 374 MB to ~125 MB.

Also copies config.json out of the HF cache. That is all the kernel needs to
instantiate the architecture - the WEIGHTS come from our checkpoint, so the
kernel never touches HuggingFace and the DINOv3 gating is irrelevant.
"""
import argparse
import glob
import json
import os
import shutil
import torch

CKPT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints"
OUT_DIR = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\kaggle_bundle"
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


def find_config(model_id):
    """Config for whichever backbone the checkpoint was trained with. A ViT-B
    checkpoint needs ViT-B's config; loading it under ViT-S's would build a
    384-dim model and fail on the state dict."""
    slug = "models--" + model_id.replace("/", "--")
    pat = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                       slug, "snapshots", "*", "config.json")
    hits = glob.glob(pat)
    if not hits:
        raise FileNotFoundError(f"config.json not found under {pat}")
    return hits[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="fullft_lr0.0001_g0.85_h1024_s2.pt",
                   help="seed 2 by default: highest VAL of the three seed replicates, "
                        "which is the clean selection criterion (they tie on test)")
    p.add_argument("--out-name", default="model.pt",
                   help="filename inside the bundle, so several models can live in "
                        "one Kaggle dataset and be switched by editing one line")
    p.add_argument("--out-dir", default=OUT_DIR,
                   help="separate dir for a ViT-B bundle, so it can be uploaded as its "
                        "own small Kaggle dataset instead of re-uploading the 1.2 GB one")
    a = p.parse_args()

    out_dir = a.out_dir
    os.makedirs(out_dir, exist_ok=True)
    ck = torch.load(f"{CKPT_DIR}/{a.ckpt}", map_location="cpu", weights_only=False)
    model_id = ck.get("model_id") or MODEL_ID

    slim = {"backbone": ck["backbone"], "head": ck["head"],
            "hidden": ck["hidden"], "tap_layers": list(ck["tap_layers"]),
            "val_mean_dice": float(ck["val_mean_dice"]),
            "source_checkpoint": a.ckpt, "seed": ck.get("seed"),
            "best_epoch": ck.get("best_epoch"), "model_id": model_id,
            # the kernel picks sigmoid vs softmax from this; dropping it would
            # silently score a bce_dice model with softmax and not crash
            "loss": ck.get("loss", "ce_dice"),
            # the kernel builds ProgressiveHead vs HeadNet from this; dropping it
            # defaults to conv and fails on a key mismatch at load time
            "head_arch": ck.get("head_arch", "conv")}
    kind = ck.get("backbone_kind") or "dinov3"
    if kind != "dinov3":
        # adapter backbones (backbones.py): the kernel rebuilds the module from the shipped config (+ code for
        # C-RADIO) and loads THIS state dict, so the checkpoint must hold the merged weights (stage 2 / LoRA);
        # a frozen stage-1 checkpoint saved "backbone": None and is not a submission model
        assert ck["backbone"] is not None, f"{a.ckpt} has no backbone weights (frozen stage-1 checkpoint)"
        slim["backbone"] = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["backbone"].items()}
        slim["backbone_kind"] = kind
        slim["backbone_name"] = os.path.basename(os.path.normpath(ck["backbone_path"])).lower()
    out_ckpt = f"{out_dir}/{a.out_name}"
    torch.save(slim, out_ckpt)
    if kind != "dinov3":
        src = ck["backbone_path"]
        if kind == "cradio":
            # remote code + config; Kaggle flattens folders, so the config is renamed per variant and the kernel
            # rebuilds a directory around it (the .py files are identical across C-RADIO variants)
            for f in glob.glob(os.path.join(src, "*.py")):
                shutil.copy(f, out_dir)
        shutil.copy(os.path.join(src, "config.json"), f"{out_dir}/config_{slim['backbone_name']}.json")
        print(f"adapter backbone {kind}: weights fp16 in {a.out_name}, config_{slim['backbone_name']}.json"
              + (" + C-RADIO code" if kind == "cradio" else ""))

    # Kaggle FLATTENS uploaded directories, so a cfg_<short>/config.json arrives
    # as a top-level config.json and collides with the other bundle's. Disambiguate
    # by FILENAME instead; the kernel rebuilds a directory around it at runtime,
    # since AutoConfig.from_pretrained needs a directory rather than a file.
    if kind == "dinov3":
        short = model_id.split("dinov3-")[-1].split("-")[0]        # vits16 / vitb16
        shutil.copy(find_config(model_id), f"{out_dir}/config_{short}.json")

    # No top-level config.json / norm.json: when two bundles are attached to one
    # kernel those names collide, and a stray config.json can hand a 768-dim
    # config to a 384-dim checkpoint. The kernel hardcodes the ImageNet
    # constants and reads config from cfg_<short>/ only.

    before = os.path.getsize(f"{CKPT_DIR}/{a.ckpt}") / 2**20
    after = os.path.getsize(out_ckpt) / 2**20
    print(f"source      : {a.ckpt}  ({before:.1f} MB)")
    print(f"had optimizer state: {'optimizer' in ck}")
    print(f"slim {a.out_name:<14}: {after:.1f} MB   ({before - after:.0f} MB stripped)")
    print(f"val_mean_dice      : {slim['val_mean_dice']:.4f}  (seed {slim['seed']}, "
          f"best epoch {slim['best_epoch']})")
    print(f"\nbundle written to {out_dir}")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f:<16}{os.path.getsize(os.path.join(out_dir, f))/2**20:8.2f} MB")


if __name__ == "__main__":
    main()
