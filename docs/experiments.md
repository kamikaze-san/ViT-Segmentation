# Severstal steel defects with DINOv3: experiment log

What was tried, what it scored, and what we learned. The machine-readable record is
`outputs/results.json` (entries `B*`, `C*`, `D*`, `E1`-`E31`, `KAGGLE_SUBMISSIONS`); every
number below comes from there. Last updated 2026-10-28.

## Task and scoring

- **Data:** 12,568 labelled 256x1600 grey steel strips, 4 defect classes; 5,506 competition test images.
- **Our splits:** `train12k` (12,068) for training, `holdout` (500) never trained on. Early work used an 8.8k train / val / test split.
- **Metric:** mean Dice over every (image, class) pair. An empty pair scores 1 if nothing is predicted and 0 otherwise, so one false pixel zeroes the pair. About 86% of pairs are empty.
- **Kaggle:** private score is the target; public is a smaller, noisier subset. 1st place: 0.90883 private.

## Current best

| | Private | Public |
|---|---|---|
| Best ensemble, V79 | **0.90874** | 0.91567 |
| Best single model, soft-target ViT-B (V95) | **0.90695** | 0.91882 |
| 1st place | 0.90883 | 0.92124 |

- **V79:** ViT-B full fine-tune + ViT-S LoRA r16 + ViT-S LoRA r8 + ViT-S sigmoid/BCE + ViT-B LoRA warm head trained with pseudo labels. Uniform average, 3-view TTA (none / h-flip / v-flip), bilinear upsampling, pixel thresholds `[.65, .60, .65, .58]`, min area `[630, 1470, 1050, 3150]`.
- **Soft ViT-B:** ViT-B LoRA r16 (all 6 linear layers), 10-epoch frozen warm head, soft targets, no class weights, round-1 pseudo labels. Thresholds `[.51, .43, .44, .32]` / `[1050, 2450, 1750, 5250]`.

**Caveat:** about 70 submissions were judged on the private score, so the best private numbers carry selection luck. Had we selected by public score, as competitors had to, our best (V83, public 0.91824) would score 0.90811 private.

## The pipeline

1. **Backbone:** DINOv3 ViT-S/16 or ViT-B/16 (Meta's weights, distilled from their 7B model), fine-tuned with LoRA rank 16 on all six linear layers per block (`q,k,v,o,up,down`), alpha 32. Most members are LoRA; one V79 member (the original ViT-B) is a full fine-tune. Once LoRA covers all six layers it matches full fine-tuning (E10: ViT-S, 0.687 full vs 0.689-0.691 LoRA at ranks 4-32), so the earlier "full FT beats LoRA" result (B6 vs B7/B9) was an artefact of LoRA touching only q/k/v, about a quarter of each block's weights (E6).
2. **Head (`HeadNet`, `spatial_head.py`):** reads blocks 2, 5, 8, 11 (LayerNorm'd, concatenated), then 1x1 conv, 3x3 conv, 1x1 conv to 80 channels, then PixelShuffle to a stride-4 class map. About 11-13M parameters.
3. **Warm head (LP-FT):** the head is first trained on the *frozen* backbone, then backbone and head train jointly from that head.
4. **Training:** CE + soft Dice (or per-class sigmoid + BCE), blackout augmentation (defects erased and filled with clean steel, label follows), 10 epochs, cosine LR, layer-wise LR decay 0.85.
5. **Targets (since E25):** per-cell class fractions at stride 4 (`--target soft`), no class weights (`--weight-power 0`).
6. **Inference:** 3-view TTA, average probabilities across members, bilinear upsampling to full resolution, per-class pixel threshold, then drop a class from an image if its area is below `min_area`.
7. **Kaggle:** `make_kaggle_bundle.py` writes slim checkpoints. `kaggle_ensemble_cell.py` is the submission cell; it reads `model_id`, `loss` and `head_arch` from each checkpoint, so these keys must survive bundling.

## What moved the Kaggle score (private)

**Changed the decision rule or robustness: these transferred.**

| Change | Effect |
|---|---|
| Per-class thresholds + min area (D1) | 0.87625 -> 0.89542 |
| min area x1.75 | +0.0027 |
| 3-view TTA | +0.00085 |
| Blackout augmentation | 0.89894 -> 0.89933 single |
| First ensemble (2 equal-quality members) | +0.00214 |
| Bilinear upsampling | +0.0003 to +0.0005 |
| Ensemble members on new axes (ViT-B, LoRA r8, sigmoid/BCE) | up to 0.90667 (4 members) |
| Warm head (LP-FT) | +0.00627 ViT-S, +0.00471 ViT-B |
| Tuned ensemble thresholds | +0.00045 (0.90756) |
| One pseudo-label member swapped in (E19) | +0.00118 (0.90874) |
| Soft targets, no class weights (E25, E26) | +0.00107 ViT-S, +0.00089 ViT-B, single model |

**Changed the fit: mostly did not transfer.**

| Change | Effect |
|---|---|
| Strong aug 16 -> 32 epochs | **-0.0059** (local score up, test-set firing up) |
| Lovasz phase (D2) | -0.0081; extended to 13 epochs, worse again |
| Progressive decoder (E15) | won 4.4 sigma on holdout, **lost** on Kaggle (0.89623) |
| Copy-paste augmentation (E13) | no benefit |
| Wrong taps (0, 3, 7, 11) | -0.00421 |
| Weighted ensemble ([1,1,1,1,2]) | -0.00043 |
| 2nd and 3rd pseudo-label members (E21) | -0.00037, -0.00063 |
| Simplex-ETF classifier head (E23) | -0.00024 (tie) |
| Metric-shaped loss (E25 arm B) | -0.00208 |
| Masked-feature continued pretraining (C2) | catastrophic forgetting; stopped at the gate |
| Continued pretraining with DINOv3's own objective (E27) | no damage, but frozen head -0.019 vs stock; stopped at the gate |

## Experiments since E19, in brief

- **E19, pseudo labelling.** The 0.90756 ensemble labelled the test set; kept only pairs all 5 members agree on, with clean pairs requiring max probability < 0.1 (3,402 images). The ViT-B student scored 0.90606 (+0.00182). Swapping it into the ensemble gave 0.90874, the current best.
- **E20, context.** Windowed (512 px) inference loses on holdout, and the test set is 2-8x more context-sensitive than holdout. DINOv3's RoPE needs a position fix for crops: without it, crops see 3.1x patch spacing.
- **E21, why more pseudo members hurt.** Students of one teacher converge: pairwise disagreement fell 8.59% -> 7.78%. Rate-matching the c3 surplus recovered only ~20% of the loss.
- **E22, self-training.** A member labelling its own data gave only +0.0006 (shared teacher: +0.0032), so ~4/5 of the pseudo-label gain is distillation from the stronger teacher.
- **E23, simplex-ETF head.** No difference at matched firing rates. The cosine head fires 94% more c2 at the same threshold, so rate-matching was essential.
- **E24, where the score is lost.** Best single model: train 0.9636, holdout 0.9513, Kaggle 0.9042. Overfitting is small; the drop to Kaggle is distribution shift.
  - **Holdout loss:** masks 61%, missed defects 31%, false alarms 8%.
  - **Masks overshoot** (1.1-1.6x the labelled area), mostly as regional extension, not a thin rim.
  - **Training targets were inflated by construction:** a stride-4 cell counted as defect if *any* of its 16 pixels was (labels +6-23% in area). Class weights also made false-positive pixels the cheapest error.
- **E25, unbiased training.** Soft per-cell targets and no class weights removed the overshoot (area/label 1.41 -> 0.90-1.10) and won on Kaggle (+0.00107, both splits up). Adding a per-pair presence term (arm B) improved masks but lost on presence (-0.00208).
- **E26, soft ViT-B.** Solo +0.00089, making it the best single model (0.90695). As one ensemble member it changed little.
  - **Mechanism, measured:** V79's tuned thresholds already shrink the averaged masks (area/label 1.11 / 1.41 / 1.04 / 1.18), so the soft fix overlaps with them. Reaching exact label size did not raise mask Dice.
  - **Verdict:** do not convert the other members for the ensemble.
- **Presence classifier (2026-08-28, backfilled).** MLP on pooled features. Lost on val: as a veto it caught 36% of false alarms but also 43 true detections, against a 57% break-even.
  - **Caveat:** it was judged on in-distribution data, where false alarms are ~10x rarer than on the test set. Never submitted to Kaggle.
- **E27, continued pretraining with DINOv3's objective.** DINO + iBOT + KoLeo with Gram anchoring to stock, 5 epochs on 17.5k unlabelled images. Details in `continued_pretraining.md`.
  - **It avoided C2's damage:** the linear probe stayed at stock level every epoch, and the weights moved a median 0.4%.
  - **It did not help:** a frozen soft head on the new backbone scored 0.6339 at epoch 10 vs stock 0.6529 (-0.019). Training loss was higher in all 10 epochs, and c3/c4 lower in nearly all. Stopped at the gate; the ~8 hr fine-tune was not run.
  - **Why (measured, `cpt_mech_diag.py`):** not Meta's documented patch-to-CLS collapse, which went the other way. The damage appears only at the full-strip geometry the run never trained on. There, early and middle layers mixed distant patches, and at block 5, the most defect-discriminative tap, defects stood out less (c3/c4 most). On the 224 crops it trained on, those layers were unharmed, and the output layers became more defect-sensitive. If revisited, train at the inference geometry.

- **E28, zero-shot VLMs (2026-10-27).** Can a current local VLM see these defects without training? Holdout strips; reference = an old ViT-S member (presence AUC 0.9997).

  | Model | Presence AUC (0.5 = random) | Behaviour |
  |---|---|---|
  | Qwen3-VL 4B, yes/no | 0.54 (200 strips) | says "defect" on 66% of clean, 74% of defect strips |
  | Qwen3-VL 4B, 2 example crops per class | c3 0.55, c4 0.86 (12 positives) | "no" to every type on all 200 strips |
  | Qwen3.5 4B (thinking off) | 0.43 (60 strips); c3 0.34 with examples | "yes" to nearly everything |
  | WeVisOne-VL-2B (segmentation) | - | predicts a defect on 100% of strips; mask Dice 0.31 |
  | LLMSeg-style loop on Qwen3-VL 4B | - | 27% of defect strips end with a mask (Dice 0.08), 0% of clean |
  | LocateAnything-3B (8-bit) | 0.36-0.40 | ~23 boxes per strip, 19% touch a label |

  Answers follow the prompt, not the image. Zero-shot and in-context use is ruled out; a VLM fine-tuned on our labels as a presence checker is still untested. Scripts `src/vlm_*.py`, results in `outputs/vlm/`.

- **E29, frozen encoders: the fair version of E28 (2026-10-27).** Same frozen probes for each encoder, trained on our labels, scored on all 500 holdout strips.

  | Encoder (frozen) | Patch-probe Dice c1 / c2 / c3 / c4 (mean) | Presence AUC c1 / c2 / c3 / c4 | Any-defect AUC |
  |---|---|---|---|
  | DINOv3 ViT-B/16 | 0.42 / 0.51 / 0.65 / 0.69 (0.568) | 0.905 / 0.993 / 0.970 / 0.999 | 0.949 |
  | DINOv3 ViT-L/16 | 0.43 / 0.32 / 0.65 / 0.68 (0.519) | 0.929 / 0.992 / 0.977 / 0.998 | 0.956 |
  | Qwen3-VL ViT (from WeVisOne-2B) | 0.43 / 0.46 / 0.67 / 0.68 (0.560) | 0.901 / 0.968 / 0.972 / 0.996 | 0.958 |

  - **The VLM's eyes are as good as DINOv3's** on these probes (c2 has only 11 positives, so its differences are noise). E28 failed on the language side: the model sees the defects but cannot be told what counts without training.
  - **But not more complementary than another DINO.** Combining two encoders' presence scores: ViT-B + Qwen 0.965, ViT-L + Qwen 0.969, and the control ViT-B + ViT-L 0.963. Agreement on clean strips is about 0.36 for every pair.
  - **So** a Qwen-ViT member or checker would be roughly one more good member, with the diversity of a different DINOv3 size. Scripts `src/encoder_probe.py`, `src/encoder_compare.py`; results in `outputs/encoder_probe/`.

  **Extended to seven encoders** (same probes; feature time measured locally):

  | Encoder (frozen) | Probe Dice c1 / c2 / c3 / c4 (mean) | Any-defect AUC | s/strip |
  |---|---|---|---|
  | DINOv3 ViT-B/16 | 0.42 / 0.51 / 0.65 / 0.69 (0.568) | 0.949 | 0.039 |
  | DINOv3 ViT-L/16 | 0.43 / 0.32 / 0.65 / 0.68 (0.519) | 0.956 | 0.087 |
  | DINOv3 ViT-H+/16 | 0.48 / 0.43 / 0.66 / 0.71 (0.572) | **0.965** | 0.192 |
  | Qwen3-VL ViT 300M (WeVisOne-2B) | 0.43 / 0.46 / 0.67 / 0.68 (0.560) | 0.958 | 0.087 |
  | Qwen3.8 vision encoder | 0.43 / 0.37 / 0.66 / 0.66 (0.527) | 0.957 | 0.122 |
  | **C-RADIOv4-SO400M** (431M) | 0.47 / 0.54 / 0.68 / 0.68 (**0.591**) | 0.957 | 0.110 |
  | C-RADIOv4-H | 0.50 / 0.43 / 0.69 / 0.69 (0.577) | 0.946 | 0.149 |

  - **Stronger encoders gain where we lose:** c1 Dice +0.05 to +0.08 (H+, both C-RADIOs), c3 +0.02 to +0.04. C-RADIOv4-SO400M has the best mean, DINOv3 H+ the best presence.
  - **Newer is not better for Qwen:** the Qwen3.8 encoder is below the 300M Qwen3-VL one (0.527 vs 0.560).
  - **Complementarity on the test set:** c3 decision disagreement with DINOv3-B at the ensemble's firing rate is 0.113-0.116 for the other DINO sizes and both C-RADIOs, 0.121-0.123 for the Qwen encoders. Nothing is clearly more independent than another DINO.
  - **So** the lever is a backbone upgrade under our recipe, not a VLM second opinion. Picked for E31: C-RADIOv4-SO400M and the Qwen ViT 300M (H+ ruled out as too slow to train).

- **E30, train vs test shift (2026-10-27).** Is the competition test set distinguishable from our training data, and by what?
  - **Very distinguishable:** a logistic domain classifier on frozen ViT-B pooled features separates test from train12k at AUC 0.92 (holdout vs test also 0.92).
  - **Mostly composition:** test has 3x more mostly-black strips (>60% dark: 25.8% vs 8.0%; >30% dark: 35.9% vs 19.5%), median brightness 72 vs 93, lower gradient energy. The steel itself is only slightly darker (96 vs 102 on strips with little black). Single statistics separate weakly (AUC 0.64-0.69), so the classifier also uses texture.
  - **Holdout is train-like** (median P(test) 0.14), consistent with holdout not ranking Kaggle (E12). Test-like training strips carry more defects: any-defect 0.66-0.68 in the most test-like 20% vs 0.44 in the least test-like half.
  - **The test defect rate cannot be estimated this way:** EM prior adjustment gave 0.56-0.999 per class. EM assumes p(x | class) is unchanged between train and test, which a 0.92 domain AUC breaks.
  - **Follow-up 1, a test-like validation:** importance-weighting the holdout by the domain classifier turns the holdout-Kaggle correlation across 10 models from Pearson -0.17 to +0.40 (Spearman -0.13 to +0.32; the most test-like 30% alone: Spearman +0.41), and moves the score level most of the way to Kaggle (0.907-0.930 vs Kaggle 0.891-0.901). Suggestive, not significant: n = 10 models, effective sample ~49 strips.
  - **Follow-up 2, firing by darkness:** the teacher ensemble fires on test *less* often than the train defect rate in every darkness bucket (any class: 0.43 vs 0.50, 0.28 vs 0.56, 0.58 vs 0.65, 0.47 vs 0.56 from least to most dark). No sign that false alarms concentrate on the dark strips that dominate the shift; c4 fires somewhat above the train rate on strips with little black (0.135 vs 0.094).
  - Scripts `src/shift_analysis.py`, `src/shift_followups.py`; results in `outputs/shift/`.

- **E31, backbone upgrade: C-RADIOv4-SO400M and the Qwen3-VL ViT under the soft recipe (2026-10-27/28).** Same two stages as soft ViT-B (`vitb_soft_pseudo.sh`), only the backbone changes.
  - **Infrastructure:**
    - `src/backbones.py` adds adapters. C-RADIO: `forward_intermediates`, taps 6/13/19/26 of 27 blocks, input in [0, 1]. Qwen ViT: our own patchify and merge-window unshuffle (equal to Qwen's processor to 6e-8), taps 5/11/17/23.
    - `train_full_ft.py --backbone cradio|qwenvit --backbone-path <dir>`. The DINOv3 path is unchanged (reproduces epoch-0 0.0160).
    - `eval_head.py` scores any head checkpoint with the training code's own evaluation.
    - Kaggle cell v2 (`kaggle_ensemble_cell_v2.py`): splits images across both T4s and handles each member's backbone kind and normalisation; `make_kaggle_bundle.py` packs adapter weights and code for offline use. Identical to v1 for V79 (192/192 rows on 48 local test images); adapter members load and run (`kaggle_local_test.py`).
    - The Mac mini (M4 Pro, MPS) is a second trainer via the `CLAUDE.md` handoff; about 1.8x slower than the PC on stage 1.
  - **Stage 1 (the gate):** frozen backbone, soft conv head, 10 epochs, holdout at epoch 10 against soft ViT-B. Tie band ~0.009 (seed sigma).

    | Backbone | c1 | c2 | c3 | c4 | Mean | vs ViT-B | Run |
    |---|---|---|---|---|---|---|---|
    | Soft ViT-B (reference) | 0.560 | 0.525 | 0.757 | 0.770 | 0.6529 | - | PC |
    | Qwen3-VL ViT 300M | 0.599 | 0.546 | 0.761 | 0.750 | 0.6640 | +0.011 | PC, 4.9 h |
    | C-RADIOv4-SO400M | 0.640 | 0.589 | 0.770 | 0.764 | 0.6907 | **+0.038** | Mac, 11.1 h; scored 0.6906 on the PC |

    Both pass. C-RADIO clearly (about 4 seed sigmas), with the gain on the small classes c1 and c2. One seed each; a seed-1 repeat of the C-RADIO gate is running on the Mac.
  - **Stage 2, C-RADIO on the PC:** LoRA r16 a32 on qkv/proj/fc1/fc2 of all 27 blocks (7.7M parameters), from the epoch-10 stage-1 head, train12k + 3,402 round-1 pseudo labels, 10 epochs.
    - **Gradient checkpointing is required:** without it the 8 GB card fills (7.9 GiB) and spills into shared system memory, more than 10x slower. With it: 0.575 s/strip, ~2.2 h per epoch, peak 3.8-5.1 GiB.
    - **Run 1 fell instead of rising:**

      | Holdout mean | ep 0 | ep 1 | ep 2 | ep 3 |
      |---|---|---|---|---|
      | C-RADIO stage 2, run 1 | 0.6906 | 0.5954 | 0.6357 | 0.6126 |
      | Soft ViT-B stage 2 | 0.6529 | 0.6435 | 0.6445 | 0.6649 |

      Training loss was above ViT-B's every epoch, and the losses were concentrated on c1/c2 (c2 0.589 -> 0.333 at epoch 1).
    - **Cause: C-RADIO changes its position embedding in training mode.** Stage 1 keeps the frozen backbone in eval mode; stage 2 calls `model.train()`. That switches C-RADIO's CPE patch generator (`vit_patch_generator.py`) to a training branch: per sample, the strip is mapped onto a random window of the 128x128 position grid (scale 0.32-1, aspect 3/4-4/3, random offset), and 10% of samples get no position embedding. Eval and Kaggle use one fixed layout: the grid resized to 100x100, strip rows in its top 16%. Even at full scale, training spreads the 16 rows over the whole grid height, so rows are 2-6x further apart than at eval. That is a systematic shift for our 1:6.25 strips, not jitter around the eval layout. Run 1 spent its epochs learning positions inference never uses, and its head (trained on eval-layout features) met a different layout at once.
    - **Measured** (`check_cradio_train_mode.py`, 4 strips, stock backbone): in training mode the tap features' cosine to the eval features is 0.77-0.97 per tap, different on every call. With the fix they are identical (max difference 0).
    - **Fix:** `RadioBackbone.train_mode_fix()` puts the patch generator back in eval mode after `model.train()`; `train_full_ft.py` calls it every epoch.
    - **Others checked:** HF DINOv3 also changes positions in training mode, but mildly: one random RoPE rescale in [0.5, 2] per batch, same for both axes, centred on eval. All earlier DINOv3 stage-2 runs had it. The Qwen ViT has no training-only behaviour (attention dropout hard-coded 0). Not affected: both stage-1 gates and all Kaggle inference (eval mode).
    - **Run 1 stopped after epoch 3** (best checkpoint, epoch 2, kept as `..._cradio_so_lora_warm_soft_pseudo01.pt`). Run 2 with the fix started 10-28 15:49 (`src/cradio_stage2_posfix.sh`, tag `_cradio_so_lora_warm_soft_posfix_pseudo01`), ~22 h.
    - **The bug was real but did not cause the drop.** Predicted: run 2's epoch 1 stays near 0.69. Got 0.5642 (c1 0.539, c2 0.255, c3 0.748, c4 0.715), below run 1's 0.5954; training loss slightly lower (0.1544 vs 0.1587). The fix stays, since it makes training match inference.
    - **Correction on the schedule:** `--gamma 0.85` is layer-wise decay and does not apply to LoRA. The LR is cosine over 10 epochs, so epochs 1-3 run at 100 / 98 / 91% of peak and epochs 8-10 at 21 / 10 / 2%. Mid-run holdout values say little about the end (soft ViT-B: 0.6366 at epoch 4, 0.7020 at epoch 10), so run 2 is judged at epoch 10 and on Kaggle. Stopping run 1 at epoch 3 was justified by the bug, not by its numbers.
    - **Open hypothesis for the drop:** LoRA at peak LR distorts C-RADIO's features much more than ViT-B's (feature distortion, as in LP-FT), hitting the small classes first. Test when the GPU is free: per-tap feature change from stock after stage 2, C-RADIO vs ViT-B.
  - **Mac follow-up checks (2026-10-28, `src/radio_checks.sh all`, results in `outputs/radio_layout_probe/results.json` and `outputs/lr_drift/summary.json`):**
    - **Layout: keep `top`.** Frozen probes, mean Dice (3 probe seeds): top 0.5936, native 0.5823 (-0.011, CI -0.036 to +0.007), center 0.5673 (-0.026, CI -0.039 to -0.017), stretch 0.5479 (-0.046, CI -0.077 to -0.008). No layout beats top by 0.01 with a CI above 0, so no new stage-1 head is needed. Center loses c2 (0.455 vs 0.550); apart from c2 it is level with top.
    - **Drift: C-RADIO's feature movement scales with LR.** 1 epoch (300 steps, 1,200 strips) at constant peak LR from the stage-1 head; mean relative change of the tap features from stock, and holdout:

      | LR | Feature drift (rel, mean of taps) | Holdout ep 0 -> 1 | Drop |
      |---|---|---|---|
      | 1e-4 | 0.200 | 0.6907 -> 0.6585 | -0.032 |
      | 3e-5 | 0.072 | 0.6907 -> 0.6763 | -0.014 |
      | 1e-5 | 0.029 | 0.6907 -> 0.6756 | -0.015 |

    - **Reading it:** 1e-4 moves the features ~7x more than 1e-5 and loses most; 3e-5 and 1e-5 tie (differences below the 0.009 seed sigma). A small drop remains even at 1e-5, so drift is not the whole story. The hypothesis was C-RADIO vs ViT-B at the same LR; the **ViT-B reference arm was not run** (head and gated weights absent on the Mac), so that comparison is still open. The short runs also did not reproduce the PC's c2 collapse (c2 0.58-0.61 here vs 0.255 in a full epoch at 1e-4): 300 steps has limited power. Candidate for a stage-2 rerun: C-RADIO LoRA at 3e-5 or lower. Single seed.
    - **Caveats (PC review):** only the backbone LR varied; the head trained at 3e-4 on 1,200 strips in every arm, so part of the ~0.015 floor is likely the head adapting to a small subset, leaving ~0.017 (about 2 seed sigmas) attributable to backbone LR at 1e-4. The drop grows with steps (300 steps: -0.032; a full epoch with pseudo labels: -0.13), so these runs show only its start. A lower LR shrinks the early dip but may not finish higher: on ViT-S, 1e-4 dipped early and still beat 5e-5 at the end (B7). So no LR change before run 2's epoch 10 and the ViT-B reference arm.
  - **Next:** Kaggle solo submission of the fixed C-RADIO with rate-matched thresholds against 0.90695 (Kaggle decides, not holdout); then Qwen stage 2.

**Not yet run:** ideas with their reasoning and cheapest tests are in `ideas_backlog.md`; open continued-pretraining hypotheses are in `continued_pretraining.md`.

## Methodology lessons

1. **Holdout cannot rank models for Kaggle.** Across 8 models, holdout and private score are uncorrelated (r = -0.12, E12), and the ordering has inverted repeatedly. Use holdout for *mechanism* checks (mask size, overshoot), not selection.
2. **Holdout also biases decision rules toward firing.** It has ~0.9% false-alarm pairs against ~9% on the test set, so a holdout-tuned threshold always says "fire more" (E21).
3. **Noise floor.** Seed sigma is about 0.009 on holdout. On Kaggle, single-run differences of a few 1e-4 are ties, and private and public moving in opposite directions is the usual sign of no real change.
4. **Compare at matched firing rates.** Any change to model, loss or ensemble shifts the probability scale. Rate-match thresholds on the test set, label-free (`*_rate_match.py`), before comparing.
5. **Verify the mechanism, then let Kaggle decide.** E24 -> E25 -> E26 is the template: measure where the loss is, change the cause, check the mechanism moved, then submit.
6. **Position geometry is a recurring trap.** Our strips are 1:6.25, and every place a model's position encoding is built for the input (E20 crops, E27 continued pretraining, E31 C-RADIO's training branch) has bitten. Before fine-tuning a new backbone, grep its code for `self.training` branches: `model.train()` can change the inputs, not just dropout.
7. **Operational lessons:**
   - Bundling must keep `loss` and `head_arch`; two silent bugs came from dropped keys.
   - Never resume from a snapshot without optimizer state, or a merged LoRA checkpoint.
   - Background jobs here are capped at 30 minutes, so long runs are launched detached (`nohup`) and watched through their logs.
   - After a wrapper is killed, check for surviving launchers before restarting anything: a duplicate stage once started this way.
   - Never edit a shell script while bash is running it (bash reads scripts as it goes); copy it instead.
   - The PC has 16 GB RAM with 1-3 GB free: a local test holding 6 models crashed a running gate (E31). Nothing memory-heavy next to a long job.

## Key scripts (`src/`)

| Script | Purpose |
|---|---|
| `train_full_ft.py` | Training. Main flags: `--backbone dinov3\|cradio\|qwenvit`, `--backbone-path`, `--model-id`, `--lora-r/--lora-alpha/--lora-targets`, `--grad-checkpoint`, `--freeze-backbone`, `--init-head`, `--backbone-ckpt`, `--head-arch conv\|progressive\|simplex`, `--target any\|soft`, `--loss ce_dice\|bce_dice\|lovasz\|metric`, `--weight-power`, `--pseudo-ids`, `--pseudo-mask-dir`, `--augment --aug-strength blackout` |
| `make_kaggle_bundle.py`, `kaggle_ensemble_cell.py` | Kaggle packaging and the submission cell |
| `pseudo_teacher.py`, `build_pseudo_labels.py`, `self_teacher.py` | Pseudo labels from an ensemble, or from a member's own TTA views |
| `*_rate_match.py`, `soft_member_match.py` | Label-free threshold matching on the test set |
| `boundary_diag.py`, `component_diag.py`, `ensemble_mask_check.py`, `mask_agreement.py` | Where the score is lost: mask size, overshoot, agreement |
| `context_diag.py`, `pseudo_diag.py` | Context use; ensemble similarity and threshold fit |
| `cpt_dino.py`, `cpt_mech_diag.py` | Continued pretraining (E27) and its mechanism diagnosis |
| `vlm_*.py` | Zero-shot VLM arms (E28) |
| `encoder_probe.py`, `encoder_compare.py`, `encoder_test_compare.py` | Frozen-encoder probes and complementarity (E29) |
| `shift_analysis.py`, `shift_followups.py` | Train-test shift and test-like validation (E30) |
| `backbones.py`, `eval_head.py`, `check_cradio_train_mode.py` | Non-DINOv3 backbone adapters, head scoring, the C-RADIO train-mode check (E31) |
| `kaggle_ensemble_cell_v2.py`, `kaggle_local_test.py` | Two-GPU, multi-backbone submission cell and its local dry run |
| `vitb_warm.sh`, `vitb_pseudo.sh`, `calib_ab.sh`, `vitb_soft_pseudo.sh`, `qwen_gate.sh`, `cradio_stage2_posfix.sh`, ... | Launchers that reproduce each experiment exactly |
