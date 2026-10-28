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

## Current plan: backbone upgrade gates
Status 2026-10-28: both gates PASSED (E31: C-RADIO +0.038 on the Mac, reproduced 0.6906 on the PC; Qwen +0.011 on the PC).
C-RADIO stage 2 run 1 (`src/cradio_stage2.sh`, 10-28 08:56) stopped after epoch 3: 0.5954 / 0.6357 / 0.6126
(ViT-B: 0.6435 / 0.6445 / 0.6649). Cause: C-RADIO train-mode position augmentation (see lessons; docs/experiments.md
E31). Run 2 with the fix running on the PC since 10-28 15:49 (`src/cradio_stage2_posfix.sh --grad-checkpoint`, log
`outputs/logs/cradio_stage2_posfix.log`, ~22 h). Run 2 epoch 1 = 0.5642: the fix did not remove the early drop.
The LR is cosine (epochs 1-3 at ~peak), so judge stage 2 at epoch 10 and on Kaggle, not mid-run. Mac: seed-1 repeat of the C-RADIO stage-1 gate; if it falls back near 0.65,
stop PC stage 2.
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

## Next on the Mac (2026-10-28): two C-RADIO checks, `src/radio_checks.sh`
Run after the seed-1 stage-1 gate finishes (not alongside it). The user approved both jobs. Nothing here trains a
model to keep: the short runs exist only to measure.
1. **Merge first.** The previous push (`finna`, 10-27 18:32) predates the adapter code, so this Mac's `backbones.py`
   and adapter-enabled `train_full_ft.py` did not come from git, and `git pull` will refuse or conflict. Move the local
   copies aside, pull, then take the PC's `backbones.py` and `train_full_ft.py` as the base and re-apply only this
   Mac's device/path edits (also in `train_lora.py` if changed). The new scripts rely on the PC's adapter API
   (`build_backbone`, `.taps`, `.norm`, `.dim`, `.module`, `.pos_gen`, `.train_mode_fix()`) and on `train_full_ft`'s
   `ADAPTER` / `TAP_LAYERS` / `tap_features` / `read_batch`. The position fix itself: `RadioBackbone.train_mode_fix` in
   `backbones.py`, plus after `model.train()` in `train_full_ft.py`: `if ADAPTER is not None: ADAPTER.train_mode_fix()`.
   Check with `bash radio_checks.sh fix`: it refuses without the fix, then `check_cradio_train_mode.py` must print
   PASS (train-mode features identical to eval with the fix).
2. **Optional ViT-B reference arm:** copy `fullft_lr0.0001_g0.85_h1024_vitb_soft_fh10_ep10.pt` (393 MB) from the PC into
   CKPT_DIR, and have DINOv3 ViT-B in the HF cache (gated: the user's login). Without both, that arm is skipped.
3. If this machine's `train_full_ft.py` needs a device flag, pass it with `EXTRA_ARGS="..."`.
4. Run: `cd src && nohup bash radio_checks.sh all > ../outputs/logs/radio_checks.log 2>&1 &`
   (or `fix` / `layout` / `drift` alone). Rough Mac time: layout ~1 h, drift ~2-2.5 h (3 C-RADIO arms + ViT-B).
- **Job 1, layout** (`cradio_layout_probe.py`, -> `outputs/radio_layout_probe/results.json`): C-RADIO's square
  position grid is laid over a strip as the top band of a 1600x1600 image by default. Compares top / center /
  stretch / native with E29's frozen probes (3 probe seeds, paired bootstrap). Switch only if a layout beats top by
  >= 0.01 mean Dice with a 95% interval above 0 (switching needs a new stage-1 head).
- **Job 2, drift** (`feature_drift.py`, -> `outputs/lr_drift/summary.json`): 1-epoch runs at constant peak LR from the
  stage-1 head on the first 1,200 labelled strips, C-RADIO at 1e-4 / 3e-5 / 1e-5 and ViT-B at 1e-4. Reports each
  run's holdout drop (epoch 0 -> 1) and how far its tap features moved from stock. Hypothesis under test: at the same
  LR, LoRA moves C-RADIO's features much more than ViT-B's; if so, the C-RADIO LR with ViT-B's drift is the one to use.
- Commit the two JSONs (logs are git-ignored) and push. Delete the `_lrdrift_*` checkpoints afterwards (~2 GB each).

## Done on the PC (2026-10-27): backbone adapters + smoke tests
- `src/backbones.py` adds `RadioBackbone` and `QwenViTBackbone`; `train_full_ft.py` takes
  `--backbone cradio|qwenvit --backbone-path <folder>` (frozen or LoRA only). The DINOv3 path is untouched: the
  refactored script reproduces the soft ViT-B stage-1 epoch-0 holdout score exactly (0.0160). Previous version:
  `src/train_full_ft_pre_backbones.py`. Frozen adapter runs save the head only (`"backbone": None`).
- Smoke results (`src/smoke_backbones.sh`, log `outputs/logs/smoke_backbones.log`), RTX 4060 Laptop, bs 4:

  | | Frozen head (stage 1) | LoRA r16 + grad checkpoint (stage 2) |
  |---|---|---|
  | C-RADIOv4-SO400M | 0.21 s/strip, peak 2.2 GiB -> 10 epochs ~7 h | 0.78 s/strip, peak 3.8 GiB, 7.70M LoRA params |
  | Qwen ViT 300M | 0.19 s/strip, peak 2.1 GiB -> 10 epochs ~6.4 h | 0.63 s/strip, peak 3.5 GiB, 6.29M LoRA params |
  | (ViT-B reference) | ~0.10 s/strip, 3.3 h | ~0.20 s/strip, ~8 h |

  Qwen patchify in torch equals Qwen's processor (max diff 6e-8). Stage 2 would take ~25-30 h per backbone on
  this GPU. Grad checkpointing is NOT droppable for C-RADIO LoRA: without it the 8 GB card fills (7.9 GiB), spills
  into shared system memory and runs >10x slower. With it: 0.575 s/strip, peak 3.82 GiB.
- Still to do: Kaggle cell support for the adapters (offline C-RADIO code + timm; transformers with Qwen3-VL),
  and the Mac items below (paths, MPS).

## Code state: what must be done on the Mac
`src/train_full_ft.py` and `src/train_lora.py` assume CUDA and Windows paths.
1. **Paths:** `train_lora.py` hardcodes `DATA_DIR` (raw competition data: `train_images/`, `test_images/`,
   `train.csv`), `MASK_DIR`, `SPLIT_DIR`, `CKPT_DIR`. Make them environment-overridable. Several scripts also
   hardcode `ROOT`/`OUT` Windows paths.
2. **Device:** CUDA-only calls (`.cuda()`, `torch.autocast("cuda", ...)`) -> pick cuda / mps / cpu. On MPS use fp16
   or bf16 autocast; set `PYTORCH_ENABLE_MPS_FALLBACK=1` and watch for slow CPU fallbacks. No bitsandbytes on Mac.
3. **Backbone adapters: done on the PC (see above).** Notes kept for reference:
   - C-RADIO: `AutoModel.from_pretrained(folder, trust_remote_code=True)` (needs `timm`); input in [0, 1], the model
     normalises itself; features via `model.radio_model.forward_intermediates(x, indices=taps, output_fmt="NLC",
     intermediates_only=True)`; 27 blocks -> taps (6, 13, 19, 26) (`taps_for()` scales our 2/5/8/11);
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
- The 5,506 competition test ids are simply the files in the raw `test_images/` folder. `data/processed/test_ids.csv`
  is NOT that: it is the early internal test split (1,886 training images, all inside train12k + holdout).
- Weights: `hf download nvidia/C-RADIOv4-SO400M`, `hf download tencent/WeVisOne-VL-2B` (Qwen ViT is inside; only
  the vision tower is needed). DINOv3 repos are gated (needs the user's HF login).

## Lessons that cost time before
- The PC has 16 GB RAM with usually only 1-3 GB free: never run a memory-heavy test (several large models loaded on
  CPU) next to a long job. On 10-27 a local Kaggle-cell test holding 6 models crashed the running Qwen gate and the
  shift analysis with `ArrayMemoryError`.
- A CUDA out-of-memory raised as `AcceleratorError` poisons the process: exit and restart, do not continue.
- Long jobs run detached (`nohup`), watched through log files; check for surviving processes before restarting
  anything (a duplicate stage once started this way).
- Bundling for Kaggle must keep `loss` and `head_arch` keys.
- Frozen-feature gains shrink after fine-tuning (ViT-B over ViT-S: +0.020 frozen -> +0.0036 on Kaggle with warm head).
- `model.train()` can change a backbone's inputs, not just dropout. C-RADIO's CPE patch generator randomly crops and
  stretches its position grid (and drops it for 10% of samples) in train mode; on our 16x100 strips the train geometry
  differs systematically from eval's. Stage 1 never sees this (frozen = eval mode), stage 2 does: run 1 of C-RADIO
  stage 2 fell 0.6906 -> 0.5954 at epoch 1. Fixed by `RadioBackbone.train_mode_fix()`. Before fine-tuning any new
  backbone, grep its code for `self.training` branches. (DINOv3 HF: mild RoPE rescale [0.5, 2] in train mode.)
- Never edit a shell script while bash is running it (bash reads scripts incrementally); copy it instead.
