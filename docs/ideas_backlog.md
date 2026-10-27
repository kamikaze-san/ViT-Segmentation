# Idea backlog

Ideas discussed but not yet run, with the reasoning behind each verdict and the cheapest test that would
settle it. Started 2026-10-27. Results go to `experiments.md` / `outputs/results.json` once run.

**Where the score is lost (E24, the basis for most priorities):** on holdout, masks 61%, missed defects
31%, false alarms 8%. Estimated on test (inferred from the holdout-to-test gap, test labels are hidden):
masks ~0.03, presence ~0.065. Soft targets (E25/E26) fixed mask size; presence is largely untouched.
Status: best ensemble V79 0.90874 private, best single 0.90695, 1st place 0.90883.

**Constraints (updated 2026-10-27):** this is a finished competition used as an experiment dataset, so any local model is fine. The binding limits are the local 8 GB GPU and time. Kaggle's 60-minute kernel limit only matters if predictions must be computed inside the notebook rather than precomputed locally.

## Ranked

_Updated 2026-10-27: the zero-shot VLM route is closed (E28); the fine-tuned checker in #2 is still open._

| # | Idea | Targets | Cheapest test | Cost | Verdict so far |
|---|---|---|---|---|---|
| 1 | **Vote-based presence rule:** fire a class only if at least k of V79's 5 members fire on their own | false alarms | Label-free rate-match on test, 1-2 submissions | no training | Best first step; decision-rule changes are what has transferred here |
| 2 | **Presence checker as a veto**, independent of the segmenters (different backbone or pretraining) | false alarms | Train on train12k image-level labels, validate on holdout with test-rate rescaling (below), then Kaggle | ~2-3 h + submissions | Highest potential. August classifier was judged on val (~10x fewer false alarms than test) with the segmenter's own backbone, never on Kaggle |
| 3 | **Round-2 pseudo labels from V79** for the existing pseudo slot only | everything, via a stronger teacher | Retrain that one member | ~26 min + ~5.5 h | Mechanism measured (E22: gain comes from teacher strength); keep to one member (E21) |
| 4 | **Decision-rule details:** drop small connected components instead of whole-class min area; 4-view TTA | masks + presence | Rate-match + submission | cheap | Small (~1e-4) but this kind has transferred |
| 5 | **Distil V79 into one soft ViT-B** | best single model | Train + submit | ~8 h | Raises the single-model ceiling; unlikely to beat the ensemble |
| 6 | Research: strip-size CPT (`continued_pretraining.md` H-A..H-D), 7B-teacher distillation, ViT-L full recipe | backbone | see that doc | high | Low odds for score; ViT-L alone takes ~52 of Kaggle's 60 min |

Kaggle's 60-minute limit applies to #2 and #4: measure V79's current runtime before adding anything.

## From outside ideas (X posts, 2026-10-27)

### Zero-shot open-vocabulary segmenters (Falcon-Perception, SAM with prompts, VLM + SAM)
- **Facts:** Falcon-Perception is 0.6B, text prompt -> instances with full-resolution masks, long side <= 1024 px (our strips shrink 1.6x or need tiling), Apache-2.0, no fine-tuning recipe published. Its own listed weakness is presence calibration (MCC 0.64 vs SAM 3's 0.82).
- **Predicted to fail here:** our classes are unnamed annotation conventions; ~86% of pairs are empty (an all-empty submission scores ~0.86), so false alarms cost immediately; presence is its weakest point; our labels are coarse, so sharp true-edge masks are not closer to them (E24: 13-32% of boundary pixels within 2 px). Corn seeds and cells are high-contrast countable objects, the easy regime.
- **Test:** zero-shot on ~300 holdout strips with a few prompts, per-class presence AUC. ~30 min. Expected poor; run only to check the prediction.
- **Result (E28, 2026-10-27): ruled out.** Six zero-shot arms (Qwen3-VL 4B with and without example crops, Qwen3.5 4B, WeVisOne-VL-2B, an LLMSeg-style loop, LocateAnything-3B): presence AUC 0.36-0.54 against 0.9997 for an old ViT-S member. Answers follow prompt framing, not the image. Details in `experiments.md` E28. What remains open is a VLM *fine-tuned* on our labels (Jev-style checker), not zero-shot use.

### Zoom-and-recheck presence checker (from the Qwen3-VL cell-labelling agent post)
- **Idea:** V79 proposes; only pairs near the decision threshold (~5-10%) go to a checker that sees a zoomed crop around the candidate region and says keep or drop. Small c1/c2 defects get more pixels. A small VLM (Qwen3-VL-2B, or Falcon) fine-tuned on train12k's image-level labels gives errors independent of DINOv3's.
- **Validation with labels:** holdout is the only clean labelled set (members trained on train12k). V79 makes ~50 errors in 2,000 holdout pairs, so it is noisy. Rescale before judging: each vetoed false alarm gains 1 pair, each wrongly dropped detection loses ~0.75 (typical TP Dice); count both at the test false-alarm rate (~9% of empty pairs vs ~0.9% on holdout). Kaggle decides.
- **Gate before training:** independence check. Linear presence probe on the VLM encoder's features; if its holdout errors mostly overlap V79's, a gate cannot help.

### RL with a pixel-level reward
- **For the DINO segmenter: not recommended.** Supervised per-pixel loss already gives the exact gradient for every pixel; RL swaps it for a noisy sampled estimate. It pays only when the objective is non-differentiable and the surrogate badly misaligned; our one cliff (presence) was tried as a differentiable term (E25 arm B, -0.00208). The train-to-Kaggle gap (0.964 -> 0.904) is distribution shift, which optimising the train metric harder does not touch.
- **For a VLM checker: fits, but only if multi-step.** Reward = the competition score of the pair (keep true defect +Dice ~0.75, drop it 0; keep false alarm 0, drop it +1). For a single keep/drop decision, supervised training plus the threshold q > 1/(1+d) reaches the same optimum. RL (GRPO, as in Falcon's RL revision: dense-scene F1 72.3 -> 80.5) earns its place only if the checker decides where to zoom and how many looks to take. GRPO on Qwen3-VL-2B with LoRA is tight on 8 GB; start supervised.

### Fine-tuned SAM
- **As a backbone (main use):** a different feature profile (trained on ~1B masks, edge-focused) means different errors. But here ensemble gain tracks solo quality (E8), the most diverse non-member was neutral (E17), and weaker ConvNeXt hurt (E9). So it must be about as good alone as our members (~0.905).
- **Head:** SAM's mask decoder is promptable and class-agnostic (prompt -> 3 candidate binary masks + quality score); no 4-class output, no presence output. Options: our HeadNet on SAM encoder features (SAM-1 has 16-px patches like DINOv3; SAM-2's multi-scale features need the FPN head from E9), or keep SAM's decoder with 4 learned class tokens (SAMed). SAM 3 adds text prompts and a presence head, but its concepts are natural-language.
- **"Sharper / better at lines":** not true of original SAM. HQ-SAM documents coarse boundaries and broken thin structures (kite-line example). SAM 2's Hiera encoder has 4-px-stride early features fed to the decoder, finer than DINOv3 ViT's 16-px patches. For us, sharper boundaries are worth little (coarse labels); detecting thin or small defects (c1 spots, thin c3 streaks) could matter, which is presence again. ConvNeXt also had fine early stages and was weaker (though undertrained).
- **Literature:** no SAM paper on Severstal found; SAM-based steel defect work exists (P3-SAM few-shot strip steel; LoRA + auto-prompt SAM for industrial defects), none against a DINOv3-class model at our data scale.
- **Test (gate):** frozen-feature probe, SAM 2 vs DINOv3 ViT-B, same head recipe (as E5), plus holdout error overlap with V79 and recall on small/thin defects. ~1-3 h, no Kaggle. Fine-tune and submit only if close in quality and different in errors.
- **Box-prompted SAM decoder** (V79 gives presence + box, SAM redraws the mask): targets masks only, which soft targets already improved. Low priority.

### Polygon output (from the Qwen-Max "box + polygon as JSON" post)
- **The model itself does not fit:** an API-served Max model cannot run in Kaggle's offline kernel or on 8 GB, and sending competition test images to a third-party service raises rule and data-use questions. Its own numbers (44.8 mAP on 40 natural images, 30-35 on dense small instances, broken coordinates) are on easier images than ours.
- **The representation does fit. Measured 2026-10-27 on 1,500 labelled train12k images (regions >= 50 px):**

| Class | Regions | Vertices at 1 px tolerance, median (p90) | Area / convex hull, median | Area / rotated rectangle, median | BBox h x w, median |
|---|---|---|---|---|---|
| c1 | 770 | 4 (13) | 0.95 | 0.81 | 43 x 25 |
| c2 | 83 | 4 (8) | 0.94 | 0.87 | 161 x 19 |
| c3 | 3,284 | 5 (15) | 0.96 | 0.82 | 152 x 34 (34% full height) |
| c4 | 441 | 16 (45) | 0.88 | 0.67 | 97 x 127 |

  Annotators drew a few-corner, nearly convex polygon per region for c1-c3 (not rectangles); c4 outlines are more detailed. Our model outputs smooth probability blobs, so prediction shape and label shape differ by construction.
- **Cheap test (no training):** fit each predicted c1-c3 component with a few-vertex polygon (`approxPolyDP` to ~4-6 vertices, or the convex hull), leave c4 alone or give it a larger budget, and compare mask Dice against holdout labels before and after. Holdout is valid for this kind of mask-shape check. Caveat: this fixes shape, not extent, and E24 found mask error is mostly regional overshoot, so the gain may be small.
- **Bigger version, only if the cheap test helps:** a head that outputs polygon vertices per instance, the way annotators label (a polygon-regression head on DINOv3, or a small VLM fine-tuned to emit polygons).

### Can a strong VLM's "reasoning" help? (discussed 2026-10-27)
- **Evidence against off-the-shelf use:** MMAD (ICLR 2025, 39,672 industrial anomaly questions): GPT-4o 74.9% average, best open-source InternVL2-76B 70.75%, experts 86.65%; models do notably worse on defect questions than on object questions. And that is multiple choice, far easier than masks.
- **Evidence on fine-tuning:** Visual-RFT (RL fine-tuning of Qwen2-VL) beats SFT by a wide margin in few-shot settings (1-shot classification +24.3% vs SFT -4.3%); the gap narrows as data grows. We have thousands of labels, where supervised training already works.
- **Mechanism:** reasoning works on what the vision encoder perceives plus prior knowledge. It cannot recover texture detail the encoder missed; it can add knowledge such as "that dark band is a reflection or a roller mark, not a defect". That fits part of our false-alarm problem, with one catch: annotators also left real defects unlabelled, so "is a defect" is not "was labelled".
- **Ways it could contribute, ranked:**
  1. **Retrieval-augmented second opinion (no training for a first test):** for each borderline pair, show the VLM the zoomed candidate, our mask overlay, and k labelled training examples retrieved by DINOv3 similarity: true class-k defects and clean lookalikes that fooled the segmenter. Ask whether the candidate matches the labelled defects or the lookalikes. The annotation convention becomes in-context evidence, so the VLM compares rather than relying on knowledge of c1-c4 it does not have. Validate on holdout borderline pairs with test-rate rescaling.
  2. **Fine-tuned VLM checker** (supervised first; RL only for a multi-step zoom policy). Its value is independent errors.
  3. **Failure analysis:** have a VLM describe and cluster our holdout errors and the test pairs where members disagree (edges, lighting bands, stains, texture), to target fixes. Insight, not score directly.
- **Local fit:** Qwen3-VL 4B/8B in 4-bit runs inference on 8 GB; a 256x1600 strip is roughly 400 visual tokens; LoRA on 2B fits for fine-tuning.

### WeVisOne-VL-2B (Tencent) and LLMSeg (arXiv 2610.29409)
- **WeVisOne-VL-2B:** Qwen3-VL-based 2B model plus a 9.6M-parameter pixel decoder (box-guided query decoder with deformable cross-attention, refining from stride 4 to stride 1). Does referring segmentation and matting, and returns `None` when the target is absent, i.e. detection, presence and mask in one model. Apache-2.0; ~4.93 GB download (model.safetensors 4.88 GB + mask_decoder.pt 38 MB); inference via its own package, Transformers or vLLM. The model card gives no benchmark numbers or fine-tuning recipe and notes fine boundaries are sensitive to resolution.
- **LLMSeg:** training-free. The MLLM detects instances, traces each as closed B-spline loops inside its box, then sees an overlay of its own mask and proposes add/remove focus boxes to retrace; up to 8 passes, at most 18 calls per image. Main backend Gemini Flash; 71.5 cIoU on ReasonSeg val, "competitive with and often surpasses" SAM 3; 21 backends, but only 26-27B ones are named, so small local models are untested. No runtime reported; masks cannot leave their box; failures in rare domains "may reflect vocabulary rather than localization".
- **For us:**
  - Both measure masks directly, so they fit the zero-shot check better than yes/no: WeVisOne gives presence (`None`) and Dice in one pass; LLMSeg tests whether a small local backend can run the spline loop at all.
  - Spline and polygon outlines resemble our labels (c1-c3 are 4-5-corner polygons).
  - The overlay self-correction loop is the novel part. Applied to our own model it becomes a learned refiner that takes (image, current mask) and outputs a corrected mask; harder to validate without a larger clean labelled set.
  - The unnamed-class problem remains; level-1 example crops are the workaround.
  - LLMSeg at scale is slow: up to 18 calls per image x 5,506 test images on a local 4B. Only viable on borderline pairs.

### Qwen-3D (ECCV 2026, arXiv 2608.02980)
- **What it is (from the abstract):** a 3D vision-language model. Visual tokens get 3D rotary position embeddings, so attention works in shared scene coordinates instead of per frame; depth and camera poses combine multiple views; a query-based decoder does grounding, instance segmentation and VQA.
- **Direct use: none.** Our data are single 2D grayscale strips with no views, depth, poses or video.
- **Transferable ideas:**
  1. **Position encoding defines the world the tokens live in.** That is our E20 / E27 finding: DINOv3's RoPE coordinates are normalised per input, so a crop and the full strip live in different coordinate worlds (the cause of the CPT side effect, H-A). Giving every patch its physical strip coordinate everywhere (training crops, tiles, zoomed re-inference) is the 2D analogue. It enables cheap crop training and higher-resolution tiles without the coordinate mismatch.
  2. **Spend tokens on content, not area.** Most of a strip is clean steel. A second pass at higher resolution only around candidate regions, with consistent coordinates, is the zoom-and-recheck idea done inside our own model.
  3. **A shared world across images (speculative):** if some strips are neighbouring tiles of the same coil, neighbours could give context. Cheap CPU check: match edge columns across images for continuity.

### VLM reasoning + SAM 3 (the "SAM3 + Qwen3.5" Spaces demo)
- **Pattern:** the VLM decides what to segment, SAM 3 draws it, i.e. decision separated from delineation.
- **For us:** the decision (presence) is our bottleneck, and it is exactly the part a zero-shot VLM cannot know for unnamed classes. Fine-tuned, the same split is backlog #2 (independent presence checker + our segmenter).
- **SAM 3 exemplar prompts** ("find regions like this labelled example") suit few-shot settings; with 12k labels a supervised model already uses every example. Click segmentation is a human tool, not automated inference.

Sources: [MMAD](https://arxiv.org/abs/2410.09453), [Visual-RFT](https://arxiv.org/pdf/2503.01785), [DINOv3](https://arxiv.org/html/2508.10104v1), [Falcon-Perception](https://huggingface.co/tiiuae/Falcon-Perception),
[HQ-SAM](https://arxiv.org/pdf/2306.01567), [P3-SAM](https://arxiv.org/pdf/2609.21424),
[PConv SAM for defects](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12031572/), [CrackSAM](https://www.sciencedirect.com/science/article/abs/pii/S0950061824017148).
