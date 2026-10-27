# Continued pretraining with DINOv3's own objective (E27)

Living design note; updated as each step finishes. Code: `src/cpt_dino.py`. Record:
`outputs/results.json` -> `E27_dino_objective_continued_pretraining_2026-10-04`.

## Question

Does continuing DINOv3's self-supervised training on our steel images, before fine-tuning,
segment better than fine-tuning stock DINOv3?

- **Matched comparison:** the soft ViT-B (Kaggle 0.90695). Same fine-tune recipe (soft targets, warm head, LoRA, pseudo labels); only the starting backbone differs.
- **Prior: weak.** Test-image exposure alone was worth ~+0.0006 (E22). DINOCell saw continued-pretraining gains wash out at 130k images; we have 17.5k.

## Why not the earlier attempt (C2, masked feature prediction)

C2 used a *new* objective at lr 5e-5 for 30 epochs. Its loss looked healthy while features were destroyed: frozen probe 0.6108 against stock 0.6763, found only after 9 hours. This design changes three things:

1. **The model's own objective family:** DINO (image level) + iBOT (masked patches) + KoLeo, with Gram anchoring to a **frozen stock copy**. Gram anchoring is Meta's tool against patch-feature drift; here it anchors to the starting weights.
2. **Head warm-up on a frozen backbone first.** Meta released no DINO/iBOT heads, and random heads would otherwise push noise into the backbone.
3. **Low LR plus guards every epoch.** LR at most 1e-5 with layer-wise decay 0.9. After each epoch, a drift check and a linear probe run against stock, and the run stops itself on failure.

**One limit:** the released ViT-S/B/L were *distilled* from a frozen 7B model, so their exact training objective can't be resumed. This is the closest faithful version.

## Design

| Item | Setting |
|---|---|
| Data | train12k + 5,506 test images, unlabelled (17,574); holdout excluded (asserted in code) |
| Views | 2 overlapping 224x224 global crops + 8 local 96x96 crops near them, native pixel scale, per-view photometric jitter and flips |
| Student / teacher / anchor | trainable HF DINOv3 ViT-B / EMA teacher (0.994 -> 1.0) / frozen stock copy |
| Heads | MLP 768 -> 2048 -> 2048 -> 256 with BatchNorm, cosine against 4,096 prototypes; separate DINO and iBOT heads; teacher heads use batch statistics |
| Losses | DINO (CE over 2 global + 8 local views) + iBOT (block-masked patches, 50% of globals, ratio 0.1-0.5) + 0.1 KoLeo + Gram on blocks 2/5/8/11, unmasked patches only |
| Centering | DINO v1 EMA (softmax) centering; teacher temperature 0.04 fixed; student 0.1 |
| Optimiser | AdamW, wd 0.04; backbone LR 1e-5 x 0.9 per layer down; heads 5e-4; 10% warmup, cosine; grad clip 3 |
| Batch | 32 (peak 4.7 GB with gradient checkpointing; 64 does not fit) |
| Schedule | 2 epochs head warm-up (backbone frozen) + 5 epochs continued pretraining, ~16 min per epoch |
| Guards (every epoch) | drift: mean cosine against stock on the four tapped layers, 200 holdout images. Probe: linear patch classifier on those layers, 600 train / 200 holdout. Stop if the probe falls >0.01 below stock or any cosine < 0.80 |
| Output | `outputs/checkpoints/cpt_vitb_ep{N}.pt` with `backbone` = EMA teacher in HF layout (`train_full_ft.py --backbone-ckpt`), plus `cpt_vitb_resume.pt` |
| Carried forward | the **final** epoch, decided in advance, not picked on holdout |

## Step 0: build and tests (done)

- **Zero-step reload:** saving at step 0 and reloading through the fine-tuning path reproduces stock features exactly (difference 0.0 on all four tapped layers).
- **Failure 1, Sinkhorn centering at small batch.** With 16,384 prototypes and 16 crops, Sinkhorn spreads every target over >= K/B prototypes, so losses stuck at log K. Switched to EMA centering with K = 4,096.
- **Failure 2, temperature ramp.** I ramped the teacher temperature 0.04 -> 0.07 inside a ~1-epoch warm-up; DINO does this over ~30 epochs. Targets got *flatter* (entropy 7.4 -> 8.2). Fixed at 0.04.
- **Data check: the near-identical-images hypothesis was wrong.** DINOv3 features of steel crops differ plenty: pairwise cosine 0.40, 60% of variance between images.
- **Head check:** a random head passes the component all images share straight through (output cosine 0.53 between images). BatchNorm in the head removes it (0.21) and widens the logit spread.
- **Smoke run after the fixes** (1,000 images): DINO loss 8.12 -> 5.9, iBOT 8.16 -> 5.7, per-image teacher entropy 7.2 -> 5.2 (log K 8.32). Batch entropy stayed ~6.7-7.1, so no collapse. Feature drift after 62 steps: cosine 0.9995.
- Earlier versions are kept as `cpt_dino_v0.py` (Sinkhorn) and `cpt_dino_v1.py` (no BN, temperature ramp).

## Steps 1-2: head warm-up + continued pretraining (done)

- **Launch:** 2026-10-04 01:41, detached, PID 39040. Log: `outputs/logs/cpt_vitb.log`; per-epoch guard results in `outputs/cpt/log_vitb.json`.
- **Stock probe reference:** mean Dice 0.4137 (c1 0.2774, c2 0.0, c3 0.6695, c4 0.7079). The linear probe never finds c2 at stock, so read the per-class values, not only the mean.
- **Head warm-up (1,098 steps):** learned cleanly. DINO loss 8.3 -> 3.96, iBOT -> 3.86, per-image teacher entropy -> 2.46 with batch entropy 5.34, so no collapse.
- **Pretraining epoch 1:** losses kept falling (DINO 3.71, iBOT 3.77, Gram penalty 0.016).
- **Incident:** the disk filled while the epoch-1 checkpoint was being written, and the run crashed at 02:12.
  - **Cause:** Chrome held ~12 GB of memory, so Windows grew the page file 23.9 -> 35.2 GB and C: dropped to ~11 GB free.
  - **Damage:** epoch 1 only. `cpt_vitb_resume.pt`, written at the end of head warm-up, is intact.
  - **Fix:** `safe_save()` now writes to a temp file, checks free space first, swaps in only a complete file, and logs a warning instead of crashing. Tested. Previous version kept as `cpt_dino_v2.py`.
- **Resumed** 02:47 from `cpt_vitb_resume.pt` (step 1,098) after the user freed disk space (C: 51 GB free). Finished 04:12. The stock probe recomputed identically (0.4137), so the guard is deterministic.

### Results of steps 1-2

| Epoch | Probe mean Dice (stock 0.4137) | c1 | c3 | c4 | Drift cosine, blocks 2 / 5 / 8 / 11 |
|---|---|---|---|---|---|
| 1 | 0.4228 | 0.313 | 0.667 | 0.712 | 0.996 / 0.988 / 0.982 / 0.996 |
| 2 | 0.4242 | 0.326 | 0.681 | 0.690 | 0.977 / 0.938 / 0.926 / 0.981 |
| 3 | 0.4171 | 0.313 | 0.682 | 0.674 | 0.965 / 0.903 / 0.889 / 0.967 |
| 4 | 0.4221 | 0.327 | 0.679 | 0.682 | 0.961 / 0.892 / 0.876 / 0.962 |
| 5 | 0.4174 | 0.318 | 0.675 | 0.678 | 0.960 / 0.890 / 0.874 / 0.961 |

Stock per class: c1 0.277, c3 0.670, c4 0.708. c2 is 0 throughout, because the linear probe never finds it.

- **No sign of the C2 failure.** The probe stayed at or slightly above stock every epoch. The changes are a reshuffle (c1 up ~0.04, c4 down ~0.03) within the noise of a 200-image probe, not a clear gain.
- **Middle layers moved the most** (block 8 down to 0.874), and drift levelled off as the learning rate decayed.
- **Weights changed gently:** median 0.4% relative, most in block 11 (q/k ~2%). Side finding: `embeddings.mask_token` is all zeros in Meta's release. iBOT trained it slightly; fine-tuning never uses it.
- **Carried forward:** `cpt_vitb_ep5.pt`, the final epoch as pre-registered. It loads strictly through `--backbone-ckpt`.

## Step 3: the gate (done: FAIL)

- **Question:** do the continued-pretrained features segment better when frozen? This is the cleanest test, because fine-tuning would re-adapt the backbone anyway.
- **Recipe:** exactly stage 1 of `vitb_soft_pseudo.sh` (10-epoch soft-target conv head on a frozen ViT-B, train12k, blackout, seed 0); only `--backbone-ckpt cpt_vitb_ep5.pt` is added. Launcher: `src/cpt_gate.sh`; log: `outputs/logs/cpt_gate.log`; tag `_vitb_cpt_soft_fh10`.
- **Ran** 2026-10-04 08:47-12:19. Provenance check passed: the backbone saved with the head is bit-identical to `cpt_vitb_ep5.pt`.
- **Rule, set before the run:** compare epoch 10. Within ~0.009 of stock (seed sigma) is a tie, and a tie still goes to step 4. Clearly below stock stops E27.

### Result

Both runs use seed 0, so they see identical batches in identical order and the per-epoch differences are paired.

| Epoch | Train loss, CPT vs stock | c1 | c2 | c3 | c4 | Mean (stock / CPT) |
|---|---|---|---|---|---|---|
| 5 | +3.7% | -0.006 | -0.049 | -0.007 | -0.009 | 0.5940 / 0.5762 |
| 6 | +3.6% | -0.000 | -0.016 | -0.007 | -0.023 | 0.6137 / 0.6020 |
| 7 | +4.4% | +0.017 | +0.002 | -0.008 | -0.014 | 0.6298 / 0.6292 |
| 8 | +4.0% | +0.008 | -0.085 | -0.009 | -0.029 | 0.6567 / 0.6280 |
| 9 | +4.2% | -0.006 | -0.059 | -0.011 | -0.018 | 0.6474 / 0.6239 |
| 10 | +3.8% | -0.003 | -0.045 | -0.010 | -0.018 | **0.6529 / 0.6339** |

- **Epoch 10: -0.019**, beyond the tie band. Best against best: 0.6339 vs 0.6567 (-0.023).
- **Consistent, not one bad epoch:** training loss higher in 10/10 epochs, c3 lower in 10/10, c4 lower in 9/10, mean lower in 9/10. Epochs 5-10 average -0.017 (sd 0.010).
- **Per class it matches the step-2 probe:** c1 level or slightly up, c4 down. The features were reshuffled, not improved.

## Verdict: stop (step 4 not run)

Continued pretraining with DINOv3's own objective avoided C2's damage but made the frozen features slightly *worse* for segmentation. Step 4 (joint LoRA + pseudo labels, ~8 hr) was not run, as pre-registered.

### Why it got worse: measured (`src/cpt_mech_diag.py`, 500 holdout strips, no training)

Three explanations were checked against the DINOv3 paper first, then tested. Feature metrics use LayerNorm'd patch features, as the head reads them.

| Layer | 2 | 5 | 8 | 11 | output |
|---|---|---|---|---|---|
| Patch-to-CLS cosine, stock -> CPT | 0.069 -> 0.107 | 0.240 -> 0.196 | 0.212 -> 0.208 | 0.625 -> **0.278** | 0.351 -> **0.175** |
| Mean patch-pair cosine, **224 crops** | +0.011 | -0.011 | -0.010 | -0.032 | -0.106 |
| Mean patch-pair cosine, **full strips** | **+0.055** | **+0.043** | **+0.028** | -0.027 | -0.041 |
| Cosine 50 patches apart (full) | +0.048 | +0.033 | +0.019 | -0.028 | -0.034 |
| Between-image variance share (full) | +0.025 | +0.023 | +0.013 | -0.060 | -0.030 |
| Drift vs stock, full / crop | 0.960 / 0.976 | **0.890 / 0.959** | **0.875 / 0.910** | 0.961 / 0.954 | 0.897 / 0.881 |
| Defect-vs-own-clean-steel AUC, c3 / c4 | +0.008 / +0.006 | **-0.020 / -0.023** | -0.008 / -0.001 | +0.079 / +0.026 | +0.077 / +0.059 |

- **Meta's documented failure did not happen.** DINOv3 (Sec. 3.2, 4.1) reports dense quality falling as patch features become CLS-like. Here they became much *less* CLS-like at block 11 and the output. Refuted.
- **The damage appears only at the geometry the run never saw.** On 224 crops (the training and Gram-anchoring geometry), the middle layers did not homogenise. On full 256x1600 strips, early and middle layers did: distant patches became more alike, each patch carried more of its image's common component, and drift was much larger than on crops (block 5: 0.890 vs 0.959).
- **Block 5 is where it hurts.** It is the most defect-discriminative tap (stock AUC 0.90-0.96), and there defects stand out less from their own image's clean steel, with c3 and c4 hit most. That matches the gate.
- **The training did help where it acted directly.** At block 11 and the output, patches became more local and defects more separable (c1 AUC +0.13). Block 11 is the weakest tap for defects, though, so this did not offset block 5.
- **Plausible mechanism, untested:** RoPE coordinates are normalised to the input, so after training only on 14x14 grids, patches far apart on a 100-wide strip look "close" in coordinate units and get mixed.
- **If revisited:** train and Gram-anchor at the inference geometry (full or wide strips), as Meta's high-resolution phase does. The upside is still modest: DINOCell found continued pretraining "nearly the same" after fine-tuning on most datasets.

### Why stock DINOv3 handles the strip and the CPT model does not

Neither was trained on 256x1600 strips. The difference is the *range* each saw. DINOv3 normalises patch positions to [-1, 1] over the input, so what the model experiences is the coordinate gap between neighbouring patches. During training HF rescales the coordinate box by s in [0.5, 2] (RoPE-box jittering, paper Sec. 3.2; applied only in train mode, which our student used).

| | Grid | Token count | Neighbour gap (with jitter) |
|---|---|---|---|
| Meta main training, 256 px globals | 16x16 | 256 | 0.063-0.25 |
| Meta high-res phase, 512 / 768 px (also done for distilled ViT-B, Sec. 5.2) | 32x32 / 48x48 | 1,024 / 2,304 | **0.021**-0.125 |
| Our CPT, 224 globals / 96 locals | 14x14 / 6x6 | 196 / 36 | 0.071-0.29 / 0.17-0.67 |
| Inference strip, horizontal / vertical (no jitter) | 16x100 | 1,600 | **0.020** / 0.125 |

- Stock's training covered both the strip's token count and its horizontal neighbour gap. It wasn't tuned to the strip; the strip fell inside its range.
- Our CPT trained only on gaps of 0.071 or more and at most 196 tokens. That is 3.5x too coarse and 8x too short for the strip, and it was anchored only there. Training on one narrow setting erodes behaviour outside it, which is why Meta added a high-res phase after low-res training.

### Open hypotheses (to test)

| ID | Hypothesis | Prediction | Test | Cost |
|---|---|---|---|---|
| H-A | **Coordinate range:** CPT tuned attention to crop-sized neighbour gaps (>= 0.071), so on the strip (0.020) distant patches look close and get mixed | Crops given strip coordinates show the same broadening | **Run 2026-10-22: not supported alone** (see decoupling table). The cheap variant is therefore unlikely to fix it. | done |
| H-B | **Sequence length:** independent of coordinates, attention trained on <= 196 tokens spreads differently over 1,600 | CPT attends farther than stock on strips but not on crops | Measure mean attention distance / entropy per layer, stock vs CPT, strips vs crops | **Run 2026-10-22: supported in layers 3-4; token count alone not sufficient (decoupling test)** |
| H-C | **Weaker teacher (ceiling):** with its own moving average as teacher, CPT can at best reorganise DINOv3's features; a fixed geometry still yields a small gain at most | Even a geometry-matched CPT ties stock after fine-tuning | Only the full chain (strip-size CPT -> gate -> fine-tune -> Kaggle), or 7B-teacher distillation | 8-10 h + 8 h + submission |
| H-D | **The late-layer gain is real:** CPT made block 11 / output more defect-separable (c1 AUC +0.13) | With the geometry side effect removed, the gate beats stock | Gate after the H-A or strip-size fix | 3.5 h |

#### H-B result (`src/cpt_attn_diag.py`, 100 holdout strips, no training)

Attention reach per layer: mean attention distance divided by the distance under uniform attention (0 = fully local, 1 = uniform). Stock -> CPT:

| Layer | 1 | 3 | 4 | 5 | 6 | 7-10 | 11 |
|---|---|---|---|---|---|---|---|
| **Strips** (16x100) | 0.523 -> 0.465 | **0.411 -> 0.540** | **0.473 -> 0.553** | 0.386 -> 0.407 | 0.531 -> 0.551 | within 0.012 | 0.681 -> 0.588 |
| **Crops** (14x14) | 0.603 -> 0.564 | 0.575 -> 0.580 | 0.643 -> 0.620 | 0.652 -> 0.624 | 0.776 -> 0.747 | -0.02 to -0.05 | 0.950 -> 0.838 |

- **Supported where it matters:** in layers 3 and 4, CPT attends much farther than stock on strips (mass beyond 10 patches: 0.398 -> 0.497 in layer 3) but not on crops. That fits the strip-only mixing found at blocks 5 and 8, which those layers feed.
- **On crops CPT became more local almost everywhere,** i.e. it fitted the geometry it trained on.
- **Not the whole story:** block 2's strip mixing (+0.055 pair cosine) has no matching change in layers 0-2, so attention reach doesn't explain every effect.
- **That result cannot separate H-A from H-B:** both widen reach. So the two were decoupled (next section).

#### H-A vs H-B: decoupling coordinates from token count (`src/cpt_attn_coords.py`)

RoPE coordinates were rescaled per axis so that token count and horizontal patch spacing vary independently. CPT-minus-stock attention reach, layers 3 / 4:

| Case | Tokens | Horizontal gap | Layer 3 / 4 |
|---|---|---|---|
| Strip, own coordinates (control) | 1,600 | 0.020 | **+0.129 / +0.080** |
| Crop, own coordinates (control) | 196 | 0.143 | +0.006 / -0.023 |
| **A:** crop with strip coordinates | 196 | 0.020 | -0.006 / -0.021 |
| **B:** strip with crop coordinates | 1,600 | 0.143 | -0.051 / -0.021 |

- **Neither factor alone reproduces the effect.** The broadening needs many tokens and fine spacing together, which is the actual strip geometry. So H-A (coordinate range alone) and H-B (token count alone) are both unsupported as the sole cause.
- **Caveat on B:** its coordinates span +-7.1, far outside what either model was trained on (stock's reach even exceeds the uniform-attention value), so B is a weak test. A is cleaner.
- **Consequence for a fix:** the cheap variant (224 crops with strip coordinates) would probably not help, because the effect needs long sequences. Only pretraining at strip geometry targets the combination.

Already tested: patch-to-CLS collapse (refuted); image-specific component (present only on strips, so a consequence of the geometry issue, not of the objective itself).

**Strip-size CPT** (the fix for H-A and H-B together): main views of wide windows (e.g. 256x1024, ~1,024 tokens, partly overlapping, sometimes the full strip) plus 96-224 px squares; Gram anchoring and iBOT on the wide views. Batch ~8, about 3-4x slower per image, roughly 8-10 h including the gate (to be timed).

**Earlier reasoning, before the measurement:**
- **The SSL losses kept falling while the probe stayed flat** (DINO 3.88 -> 3.1, iBOT 3.97 -> 2.3 over 5 epochs; probe 0.417-0.424 throughout). The model got better at the objective and not at the task, so more epochs of this recipe would not have helped.
- **The image-level DINO term is at best neutral for defects.** It rewards two overlapping crops of the same strip agreeing. That agreement is driven by what differs between images (brightness, texture, strip type: ~60% of feature variance), and a defect seen in one crop but not the other is something the loss teaches the model to ignore.
- **Coverage was not the limit.** Each epoch's teacher crops cover ~17% of a strip (~62% after 5 epochs), but the probe did not improve between epoch 1 and epoch 5 as coverage rose.
- **The defect-relevant part (iBOT) sees few defects.** It masks 10-50% of patches in half the global crops, and defects are small and absent from about half the images.

**If revisited:** change the objective, not the schedule. For example, iBOT only, with masking aimed at defect regions. That borrows label information, which fine-tuning already uses more directly, so it is low priority.
