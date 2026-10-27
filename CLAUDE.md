# Severstal steel defects with DINOv3: handoff

This repo was moved from a Windows PC (RTX 4060 Laptop, 8 GB) to a Mac mini M4 Pro (48 GB unified memory,
PyTorch MPS) to run experiments in parallel. Read this first, then `docs/experiments.md` (full log),
`docs/ideas_backlog.md` (what is open and why), `docs/continued_pretraining.md` (E27), and
`outputs/results.json` (machine-readable record, entries `B*`..`E29`).

## Working rules (from the user, keep them)
- **Never start a run without asking**, even small ones. Propose the exact command and cost, wait for a yes.
- **Never kill processes you did not start** (the user runs other projects on the same machines).
- **Set the decision rule before the run** (pre-register), then report against it honestly, including failures.
- **Do not chase noise.** The user wants structural ideas that cause real growth, not hyperparameter tweaks.
  Kaggle differences of a few 1e-4 are ties.
- Document every finished experiment in `docs/experiments.md` and `outputs/results.json` (back up the JSON first).
- Do not delete user files; propose moves instead.

## Task and where we stand
- 256x1600 grey steel strips, 4 unnamed defect classes. Metric: mean Dice over (image, class) pairs; an empty pair
  scores 1 only if nothing is predicted. ~86% of pairs are empty.
- Finished Kaggle competition, used as an experiment dataset; scored by late submissions from a notebook (T4 x2,
  60-min limit; the submission cell `src/kaggle_ensemble_cell.py` currently uses only one T4).
- Best ensemble V79 **0.90874** private; best single model **soft ViT-B, 0.90695** (V95); 1st place 0.90883.
- Biggest remaining loss is train->test shift and presence errors (false alarms ~10x more frequent on test than
  holdout). **Holdout cannot rank models for Kaggle** (r = -0.12); use it for mechanism checks and gates only.
  Always rate-match thresholds on the test set (label-free) before comparing models on Kaggle.

## Current plan: backbone upgrade gates (not started)
E29 (frozen linear probes, 500 holdout strips) found two encoders worth fine-tuning under our recipe:

| Frozen encoder | Probe Dice c1/c2/c3/c4 (mean) | Any-defect AUC | Cost vs ViT-B |
|---|---|---|---|
| DINOv3 ViT-B (current backbone) | 0.418/0.514/0.647/0.693 (0.568) | 0.949 | 1.0x |
| **C-RADIOv4-SO400M** (`nvidia/C-RADIOv4-SO400M`) | 0.466/0.544/0.679/0.676 (0.591) | 0.957 | 2.8x |
| **Qwen3-VL ViT 300M** (vision tower of `tencent/WeVisOne-VL-2B`) | 0.432/0.457/0.666/0.683 (0.560) | 0.958 | 2.2x |

User decision: test these two (not DINOv3 ViT-H+, too slow to train), each against the best single model.

**Recipe = exactly the soft ViT-B one** (`src/vitb_soft_pseudo.sh`), only the backbone changes:
1. Stage 1 (the gate): frozen backbone, soft-target conv HeadNet, 10 epochs, train12k, blackout aug, seed 0,
   `--weight-power 0 --target soft --hidden 1024 --bb-lr 1e-4 --bs 4`.
   **Reference: soft ViT-B stage 1, holdout at epoch 10 = 0.6529** (c1 0.560, c2 0.525, c3 0.757, c4 0.770; best 0.6567 at ep 8).
   **Rule:** compare epoch 10. Within ~0.009 (seed sigma) = tie, a tie still proceeds; clearly below = stop that backbone.
2. Stage 2 (only if the gate passes): joint LoRA r16 a32 on all attention + MLP linears, `--gamma 0.85`,
   10 epochs, from the stage-1 head, train12k + round-1 pseudo labels (`data/processed/pseudo_ids.csv`).
3. Kaggle: solo submission with rate-matched thresholds vs 0.90695; then, for Qwen, also as a V79 member swap.

Estimated time on the PC GPU (scale from ViT-B: stage 1 3.3 h, stage 2 ~8 h): SO400M ~9 h + ~22 h,
Qwen ~7 h + ~18 h. Mac speed unknown: run a short benchmark first.

## Code state: what must be done before the gates (not done yet)
`src/train_full_ft.py` and `src/train_lora.py` assume Hugging Face DINOv3, CUDA, and Windows paths.
1. **Paths:** `train_lora.py` hardcodes `DATA_DIR` (raw competition data: `train_images/`, `test_images/`,
   `train.csv`), `MASK_DIR`, `SPLIT_DIR`, `CKPT_DIR`. Make them environment-overridable. Several scripts also
   hardcode `ROOT`/`OUT` Windows paths.
2. **Device:** CUDA-only calls (`.cuda()`, `torch.autocast("cuda", ...)`) -> pick cuda / mps / cpu. On MPS use fp16
   or bf16 autocast; set `PYTORCH_ENABLE_MPS_FALLBACK=1` and watch for slow CPU fallbacks. No bitsandbytes on Mac.
3. **Backbone adapters** (working feature extractors already exist in `src/encoder_probe.py`):
   - C-RADIO: `AutoModel.from_pretrained(folder, trust_remote_code=True)` (needs `timm`); input in [0, 1], the model
     normalises itself; features via `model.radio_model.forward_intermediates(x, indices=taps, output_fmt="NLC",
     intermediates_only=True)`; 27 blocks -> taps (6, 13, 20, 26) (`taps_for()` scales our 2/5/8/11);
     LoRA targets `qkv`, `proj`, `fc1`, `fc2` (q/k/v are packed). Check its remote code for CUDA-only paths on MPS.
   - Qwen3-VL ViT: 24 blocks x 1024, taps (5, 11, 17, 23) captured at the deepstack merger inputs; Qwen orders
     patches in 2x2 merge windows, un-shuffle to row-major (see `QwenViT` in `encoder_probe.py`); its processor
     patchifies with temporal_patch_size 2, so reproduce that in torch to keep blackout augmentation; image mean/std
     0.5. LoRA on its attention qkv/proj and MLP linears.
   - Layer-wise LR decay groups need mapping to each backbone's block names; `--head-arch conv` HeadNet takes the
     4 LayerNorm'd taps concatenated (dim = 4 x width).
   - Kaggle cell: needs the same adapters, offline (ship C-RADIO's .py files and timm; Qwen needs a transformers
     version with Qwen3-VL).
4. Smoke test each (200 images, 1 short epoch): loss falls, memory fits, checkpoint reloads in the Kaggle cell.

## Data and weights (not in git)
- Raw data: `kaggle competitions download -c severstal-steel-defect-detection` (1.7 GB; user's own Kaggle token).
- In git: split lists (`data/processed/*_ids.csv`), GT masks (`data/processed/masks`, `masks_split`), round-1 pseudo
  labels (`pseudo_ids.csv`, `pseudo_masks/`, from the V68 teacher, must not be regenerated). Not in git:
  `data/processed/images` (unused copy), checkpoints, Kaggle bundles, `external/`, caches, `.npy/.npz`.
- `outputs/pseudo/test_ids.csv` is git-ignored; use `data/processed/test_ids.csv` (same 5,506 ids).
- Weights: `hf download nvidia/C-RADIOv4-SO400M`, `hf download tencent/WeVisOne-VL-2B` (Qwen ViT is inside; only
  the vision tower is needed). DINOv3 repos are gated (needs the user's HF login).

## Lessons that cost time before
- A CUDA out-of-memory raised as `AcceleratorError` poisons the process: exit and restart, do not continue.
- Long jobs run detached (`nohup`), watched through log files; check for surviving processes before restarting
  anything (a duplicate stage once started this way).
- Bundling for Kaggle must keep `loss` and `head_arch` keys.
- Frozen-feature gains shrink after fine-tuning (ViT-B over ViT-S: +0.020 frozen -> +0.0036 on Kaggle with warm head).
