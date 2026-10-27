# Severstal steel defects with DINOv3: experiment log

What was tried, what it scored, and what we learned. The machine-readable record is
`outputs/results.json` (entries `B*`, `C*`, `D*`, `E1`-`E27`, `KAGGLE_SUBMISSIONS`); every
number below comes from there. Last updated 2026-10-04.

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

**Not yet run:** ideas with their reasoning and cheapest tests are in `ideas_backlog.md`; open continued-pretraining hypotheses are in `continued_pretraining.md`.

## Methodology lessons

1. **Holdout cannot rank models for Kaggle.** Across 8 models, holdout and private score are uncorrelated (r = -0.12, E12), and the ordering has inverted repeatedly. Use holdout for *mechanism* checks (mask size, overshoot), not selection.
2. **Holdout also biases decision rules toward firing.** It has ~0.9% false-alarm pairs against ~9% on the test set, so a holdout-tuned threshold always says "fire more" (E21).
3. **Noise floor.** Seed sigma is about 0.009 on holdout. On Kaggle, single-run differences of a few 1e-4 are ties, and private and public moving in opposite directions is the usual sign of no real change.
4. **Compare at matched firing rates.** Any change to model, loss or ensemble shifts the probability scale. Rate-match thresholds on the test set, label-free (`*_rate_match.py`), before comparing.
5. **Verify the mechanism, then let Kaggle decide.** E24 -> E25 -> E26 is the template: measure where the loss is, change the cause, check the mechanism moved, then submit.
6. **Operational lessons:**
   - Bundling must keep `loss` and `head_arch`; two silent bugs came from dropped keys.
   - Never resume from a snapshot without optimizer state, or a merged LoRA checkpoint.
   - Background jobs here are capped at 30 minutes, so long runs are launched detached (`nohup`) and watched through their logs.
   - After a wrapper is killed, check for surviving launchers before restarting anything: a duplicate stage once started this way.

## Key scripts (`src/`)

| Script | Purpose |
|---|---|
| `train_full_ft.py` | Training. Main flags: `--model-id`, `--lora-r/--lora-alpha/--lora-targets`, `--grad-checkpoint`, `--freeze-backbone`, `--init-head`, `--backbone-ckpt`, `--head-arch conv\|progressive\|simplex`, `--target any\|soft`, `--loss ce_dice\|bce_dice\|lovasz\|metric`, `--weight-power`, `--pseudo-ids`, `--pseudo-mask-dir`, `--augment --aug-strength blackout` |
| `make_kaggle_bundle.py`, `kaggle_ensemble_cell.py` | Kaggle packaging and the submission cell |
| `pseudo_teacher.py`, `build_pseudo_labels.py`, `self_teacher.py` | Pseudo labels from an ensemble, or from a member's own TTA views |
| `*_rate_match.py`, `soft_member_match.py` | Label-free threshold matching on the test set |
| `boundary_diag.py`, `component_diag.py`, `ensemble_mask_check.py`, `mask_agreement.py` | Where the score is lost: mask size, overshoot, agreement |
| `context_diag.py`, `pseudo_diag.py` | Context use; ensemble similarity and threshold fit |
| `cpt_dino.py` | Continued pretraining (E27) |
| `vitb_warm.sh`, `vitb_pseudo.sh`, `calib_ab.sh`, `vitb_soft_pseudo.sh`, ... | Launchers that reproduce each experiment exactly |
