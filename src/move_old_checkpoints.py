"""Move old checkpoints off C: to E:\\severstal_backup\\checkpoints (copy, verify size, then remove from C:).

Keeps on C: the 8 files current work needs: the soft ViT-B (best single model), its frozen warm head, and the
sources of V79's pseudo-label members. A manifest of every moved file is written next to the checkpoints.
"""
import os, shutil, time

SRC = r"C:\Users\NewGr\Downloads\ViT Project severstal\severstal\outputs\checkpoints"
DST = r"E:\severstal_backup\checkpoints"
KEEP = {
    "fullft_lr0.0001_g0.85_h1024_vitb_lora_warm_soft_pseudo01.pt",
    "fullft_lr0.0001_g0.85_h1024_vitb_lora_warm_soft_pseudo01_ep10.pt",
    "fullft_lr0.0001_g0.85_h1024_vitb_soft_fh10.pt",
    "fullft_lr0.0001_g0.85_h1024_vitb_soft_fh10_ep10.pt",
    "fullft_lr0.0001_g0.85_h1024_vitb_lora_warm_pseudo01.pt",
    "fullft_lr0.0001_g0.85_h1024_vitb_lora_warm_pseudo01_ep10.pt",
    "fullft_lr0.0001_g0.85_h1024_bce_blackout_pseudo01.pt",
    "fullft_lr0.0001_g0.85_h1024_bce_blackout_pseudo01_ep16.pt",
}

os.makedirs(DST, exist_ok=True)
files = sorted(f for f in os.listdir(SRC) if os.path.isfile(os.path.join(SRC, f)) and f not in KEEP)
manifest = os.path.join(SRC, "MOVED_TO_E_severstal_backup.txt")
t0, moved, total = time.time(), 0, 0
with open(manifest, "a") as man:
    for f in files:
        s, d = os.path.join(SRC, f), os.path.join(DST, f)
        size = os.path.getsize(s)
        shutil.copy2(s, d)
        if os.path.getsize(d) != size:
            print(f"SIZE MISMATCH, left on C: {f}", flush=True)
            continue
        os.remove(s)
        man.write(f"{f}\t{size}\t{d}\n"); man.flush()
        moved += 1; total += size
        if moved % 20 == 0:
            print(f"moved {moved}/{len(files)} | {total / 1e9:.1f} GB | {total / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
print(f"DONE moved {moved}/{len(files)} files, {total / 1e9:.1f} GB in {(time.time() - t0) / 60:.1f} min -> {DST}", flush=True)
