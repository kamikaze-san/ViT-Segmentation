"""Fold LoRA adapters into the base weights, producing an ordinary checkpoint.

LoRA freezes W and learns A, B so the effective weight is W + (alpha/r) * B @ A.
Merging computes that sum once and writes it back as a plain weight matrix. The
result is mathematically identical to the adapted model but has no adapters, so
it loads through the standard "backbone"/"head" path and the Kaggle kernel needs
no changes at all.

Why bother: the LoRA models were dropped on IN-DISTRIBUTION evidence (full FT beat
LoRA r16 by 0.0220 at 11 sigma on our test split) and never submitted. That is the
same instrument that ranked ViT-B last right before ViT-B became the backbone of
our best result. LoRA's constrained capacity is a regulariser, and regularisation
is the category that has actually transferred here (augmentation, blackout). So
the ordering may invert on the target, and r4/r8/r16 give three points on a
capacity-vs-transfer curve for the price of three submissions and no GPU training.

Matched counterpart is B7 (full FT, same 8.8k split, unaugmented) at private 0.85750.
"""
import argparse
import os
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel

from train_lora import MODEL_ID, CKPT_DIR


def merge(ckpt_name, out_name):
    ck = torch.load(f"{CKPT_DIR}/{ckpt_name}", map_location="cpu", weights_only=False)
    r, alpha = ck["rank"], ck["alpha"]
    base = AutoModel.from_pretrained(MODEL_ID)
    peft_model = get_peft_model(base, LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj"]))
    missing, unexpected = peft_model.load_state_dict(ck["lora"], strict=False)
    assert not unexpected, f"unexpected adapter keys: {unexpected[:3]}"
    n_lora_loaded = sum(1 for k in ck["lora"])
    merged = peft_model.merge_and_unload()          # W <- W + (alpha/r) B@A

    # verify the merge actually changed the weights, rather than silently no-op'ing
    fresh = AutoModel.from_pretrained(MODEL_ID).state_dict()
    msd = merged.state_dict()
    changed = sum(1 for k in msd if k in fresh and not torch.equal(msd[k], fresh[k]))
    assert changed > 0, "merge changed NOTHING - adapters did not attach"

    out = {"backbone": msd, "head": ck["head"], "hidden": ck["hidden"],
           "tap_layers": tuple(ck["tap_layers"]), "model_id": MODEL_ID,
           "val_mean_dice": float(ck["val_mean_dice"]),
           "source_checkpoint": ckpt_name, "lora_rank": r, "lora_alpha": alpha,
           "note": "LoRA adapters merged into base weights; architecturally a plain model"}
    torch.save(out, f"{CKPT_DIR}/{out_name}")
    print(f"{ckpt_name:<34} r={r:<3} adapters={n_lora_loaded:<4} "
          f"weights changed={changed:<4} val={ck['val_mean_dice']:.4f} -> {out_name}")
    return changed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=[
        "lora_r4_h1024_wp0.3_r4.pt:merged_lora_r4.pt",
        "lora_r8_h1024_wp0.3_r8.pt:merged_lora_r8.pt",
        "lora_r16_h1024_wp0.3_full.pt:merged_lora_r16.pt"])
    a = ap.parse_args()
    for pair in a.pairs:
        src, dst = pair.split(":")
        if os.path.exists(f"{CKPT_DIR}/{src}"):
            merge(src, dst)
        else:
            print(f"SKIP missing {src}")
