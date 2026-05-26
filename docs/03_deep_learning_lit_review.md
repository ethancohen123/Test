# Literature review — Deep learning for single-target person detection
# and tracking on **thermal / IR aerial drone footage** (inference only)

Scope: phase 2 of the assignment. Single-object tracking of a person in
aerial thermal footage. **Inference mode only** — no training. Hard
constraint: thermal, not RGB. Aerial drone perspective with small
target (10–30 px), partial occlusion.

Out of scope: anything that requires fine-tuning, multi-object tracking,
or RGB-only pretrained weights with no transfer story.

---

## 1. The thermal-aerial reality check

Why we cannot just plug in COCO-pretrained detectors and modern Siamese
transformers and call it done:

- **Domain gap.** A "person" in COCO is a 1080p RGB upright pedestrian
  with skin, clothing colour, faces. A "person" in our clip is a 10–30 px
  grayscale blob seen top-down. Models trained on COCO leak nearly all
  their feature-bank capacity into colour, texture, and head/torso/limb
  layout cues that simply do not exist here.
- **Top-down aerial perspective.** Most thermal-pedestrian datasets
  (FLIR ADAS, KAIST, LLVIP) are **front-facing automotive / urban**.
  Their "person" class is a vertical pedestrian a few hundred pixels
  tall. We need top-down, ~20 px.
- **Tiny target size.** SOTA thermal SOT on tiny UAVs (CST Anti-UAV,
  ICCV-W 2025) reports the *best* method achieves only **35.9 %** state
  accuracy on tiny targets vs **67.7 %** on regular-sized
  targets (Anti-UAV410). Tiny + thermal is a hard regime even for
  current SOTA.
- **Modern transformer trackers degrade.** OSTrack and MixFormer
  literature explicitly notes that they "lack the ability to track UAVs
  in TIR mode when target appearance variation, target disappearance,
  or camera movement occurs" — *exactly* our failure modes.

Practical implication: we should privilege models trained on **aerial
thermal** over models trained on RGB-then-grayscaled, and we should not
assume a generic SOT transformer will save us.

---

## 2. Datasets (so we know what pretrained weights exist)

| Dataset | Modality | Perspective | Size | Relevance |
|---|---|---|---|---|
| **HIT-UAV** | thermal | **aerial, UAV** | 2,898 imgs, "Person" class | ⭐ closest match |
| **FLIR ADAS v2 (Teledyne)** | thermal | automotive | 26,442 imgs | strong "person" prior, wrong perspective |
| **LLVIP** | visible+IR | static low-light | 15,488 pairs | weak fit (street level) |
| **KAIST Multispectral** | RGB+thermal | automotive | 95k frames | weak fit |
| **NII-CU Multispectral** | RGB+FIR | aerial | 5,880 pairs | aerial but pairs |
| **LSOTB-TIR** | thermal | mixed | 1,416 seqs / 643k frames | ⭐ SOT benchmark/training corpus |
| **Anti-UAV / Anti-UAV410** | thermal | ground-to-air | hundreds of seqs | drone-vs-drone; small-target lessons |
| **CST Anti-UAV (ICCVW 2025)** | thermal | ground-to-air | 220 seqs, 240k bboxes | tiny UAVs in clutter, hardest known |

Headline: **HIT-UAV is the only public thermal dataset whose "Person"
class genuinely matches our aerial perspective.** YOLOv8m trained on
HIT-UAV reports mAP@0.5 ≈ **0.855** on its own test split.

---

## 3. Detection methods

### 3a. Detector families to consider

| Family | Notes for thermal aerial |
|---|---|
| **YOLOv5 / YOLOv8 / YOLOv11** (single-stage) | De-facto standard. Open weights for many thermal datasets. Cheap on CPU (YOLOv8n ≈ 50 ms/frame). |
| **RT-DETR** | Transformer-based real-time detector. Slightly better accuracy at similar latency, but fewer thermal-pretrained checkpoints available. |
| **Grounding DINO / OWL-ViT** (open-vocabulary) | Zero-shot text-prompt detection ("person"). Strong on RGB; **degrades on grayscale aerial thermal** because the language–vision alignment is RGB-trained. Slow on CPU (≥ 300 ms/frame). |
| **Small-target-specific (TridentNet, SDPNet, etc.)** | More care for small targets; less open weights. |

### 3b. Pretrained checkpoints worth pulling

- **YOLOv8 + HIT-UAV** — repo and weights available (Suo et al., NPJ Sci.
  Data 2023). **Best single-shot fit.**
- **YOLOv8 + Teledyne FLIR ADAS v2** — many community repos
  (e.g. `mpolinowski/yolov8-nightshift`, `tfiroze/Thermal-Image-Object-Detection`,
  `MclarenTsang/Human-detection-in-thermal-imaging`).
- **MMTOD** (Multi-modal Thermal Object Detector, Devaguptapu et al.,
  CVPR-W 2019) — Faster R-CNN with thermal pretrained branch
  (`tdchaitanya/MMTOD`).
- **COCO-pretrained YOLOv8** with grayscale-replicated channels —
  feasible as a fallback but expected to be weaker than HIT-UAV
  weights.

### 3c. The Davis & Sharma observation that survives into 2026

Even in 2026 the classical empirical finding holds: for thermal,
**region statistics dominate edge statistics**. Detectors whose backbone
relies heavily on edge gradients (e.g. older HOG-style heads,
saliency-derived RPNs) underperform models with learned region-statistic
features. YOLO's CSP backbones are fine in practice.

---

## 4. Single-Object Tracking methods

### 4a. Two architectural philosophies

**Tracking-by-Detection (TBD).** Detect every frame, associate
across frames with a motion + appearance model (Kalman, ReID
embedding). The track is a *consequence* of the detections.
+ Simple, modular, plays well with a detector that already knows what a
  "person" is.
+ Handles full occlusions cleanly (no detection → coast / lost,
  re-acquire on next detection).
− Relies entirely on detector recall.

**Template-based SOT.** A network learns to match a template (the
target patch from the init frame) against subsequent frames. The
template itself defines the target.
+ Doesn't need a "person" class — works on any object.
+ Frame-to-frame continuity is the model's native unit.
− Needs an initial bbox.
− In TIR, template tends to drift onto warm clutter (rocks, fauna).
− State-of-the-art transformer SOT (OSTrack, MixFormer) **drops sharply**
  on TIR + tiny + clutter (CST Anti-UAV evidence).

### 4b. The candidate trackers (open code + weights)

| Tracker | Year | CPU fps* | TIR transfer | Notes |
|---|---|---|---|---|
| **SiamFC** (Bertinetto) | 2016 | ~80 | OK | smallest, most CPU-friendly Siamese |
| **SiamRPN++** (Li) | 2019 | ~20 | OK; weak on small targets | classic Siamese baseline |
| **ATOM / DiMP / PrDiMP** (Danelljan) | 2019–20 | ~10 (CPU) | very strong on TIR (LSOTB-TIR top performer family) | discriminative online learner |
| **STARK / OSTrack / MixFormer / AiATrack** | 2021–22 | ~3–8 (CPU) | weaker on TIR aerial tiny | transformer SOT |
| **LightTrack** (Yan, CVPR 2021) | 2021 | **~120** | OK | NAS-found, edge-friendly |
| **MobileTrack** | 2024 | ~150 | OK | edge-device focused |
| **Anti-UAV specific (e.g., SiamDT, RDTTrack)** | 2024–25 | ~10 | ⭐ trained for our exact regime | requires a paired modality (depth) in some variants |

*CPU fps numbers are order-of-magnitude on a modern desktop CPU at
~640 px input; treat as guidance, not promise.

### 4c. The empirically robust thermal SOT methods

From the LSOTB-TIR benchmark papers and follow-ups:
- DiMP family (online discriminative) consistently outperforms pure
  Siamese on TIR.
- Cross-modal distillation (RGB→TIR knowledge transfer; Wu et al. 2021)
  is the main reason recent TIR SOT works.
- Transformers do not yet dominate on TIR + tiny + clutter — they tend
  to overfit to the RGB visual prior.

---

## 5. Foundation models (zero-shot)

Worth noting for completeness:
- **Grounding DINO** — text-prompt detection, *"person"*. Performs well
  on visible-spectrum aerial / surveillance imagery, but its
  language–vision alignment is RGB. Slow on CPU. Useful as a sanity
  oracle to label a few frames, **not** as a per-frame detector.
- **SAM / SAM-2** — segmentation, no class concept. We could use it for
  bbox→mask refinement, but the assignment values bbox tracking.
- **F-ViTA** (CVPR 2025) — visible→thermal *translation* foundation
  model; sometimes used to augment thermal training data. Not relevant
  to inference-only.
- **CLIP / OpenCLIP image encoders** — fine for ReID embeddings, even on
  thermal (the early layers capture rough shape/contrast and survive
  the modality change reasonably well).

---

## 6. What this means for our system

### 6a. The strongest inference-only fit

```
                Thermal-pretrained                 Independent
                aerial detector                    motion evidence
                (YOLOv8 / HIT-UAV)                 (existing persistence map)
                       │                                 │
                       └──────────┬──────────────────────┘
                                  ▼
                       Candidate union + scoring
                       (person prior ∪ motion prior)
                                  │
                                  ▼
                       Existing Kalman + ReID + state machine
                       (TRACKING / COASTING / LOST)
                                  │
                                  ▼
                              Annotated output
```

Concretely:
1. **Detector block (NEW)** — YOLOv8 with HIT-UAV / FLIR-pretrained
   weights, person class only. Input: 3-channel replicated grayscale.
   Output: bbox + confidence list.
2. **Motion block (KEEP)** — our existing ego-motion compensated
   persistence detector. Same bbox+score interface. Keep as a parallel
   stream and a fallback.
3. **Fusion** — union of both candidate sets; ReID + Kalman picks the
   best per frame. When the detector misses (heavy canopy), the motion
   path can still produce a candidate.
4. **Tracking block (KEEP)** — Kalman + appearance NCC + Mahalanobis
   gate. No structural change.
5. **Optional ReID upgrade** — replace the patch-NCC appearance model
   with a CLIP image-encoder embedding (cosine similarity). More
   robust across small appearance changes.

### 6b. Why **not** template-based SOT (Siamese / Transformer) as the
core tracker

- Needs initial bbox → would need the detector anyway → at that point
  the detector + motion association is already doing the job, and
  doing it without the transformer's known TIR-tiny degradation.
- OSTrack / MixFormer drop on TIR + tiny + camera-motion (the *exact*
  shape of our clip).
- Transformer SOT on CPU is too slow to add value as a parallel safety
  net.

### 6c. Why **not** zero-shot foundation models as the core

- Grounding DINO is RGB-trained at the language level; "person" prompt
  is calibrated against RGB pedestrians, not 20-px thermal blobs.
- CPU latency is prohibitive for a near-real-time demo.

### 6d. Open code we will actually use

- `ultralytics/ultralytics` — YOLOv8 framework, ONNX/PyTorch inference.
- HIT-UAV YOLOv8 weights — pulled from the dataset repo / community
  mirrors. Fallback: FLIR-trained weights from `mpolinowski/yolov8-nightshift`
  or `MclarenTsang/Human-detection-in-thermal-imaging`.
- Optional: `IDEA-Research/GroundingDINO` for a one-shot sanity check
  on a few frames (does the model *ever* see our walker as a person?).
- Optional: `openai/CLIP` or `mlfoundations/open_clip` for ReID
  embeddings.

---

## 7. Recommended strategy

**Stage 1 (must-have for the deliverable):**
- Add a `DLDetector` block that wraps a pretrained YOLOv8 thermal model
  (HIT-UAV weights, fallback FLIR). Provide the **same `list[Detection]`
  interface** the motion detector exposes, so the tracking layer below
  it is unchanged.
- Add a new `--pipeline dl` flag alongside the existing
  `--pipeline motion` and `--pipeline intensity`, so the same script
  produces directly comparable annotated videos for the report.

**Stage 2 (nice-to-have, single small additional block):**
- A `--pipeline hybrid` mode that unions the YOLO and motion candidate
  sets and lets the tracker pick. Direct ablation against `dl` and
  `motion` makes the deliverable's failure-mode-analysis section much
  cleaner.

**Stage 3 (only if Stage 1 alone is unsatisfactory):**
- Swap the patch-NCC ReID for a CLIP image-encoder embedding.

**What we are explicitly NOT doing:**
- No training, no fine-tuning. The assignment is inference-only.
- No transformer SOT tracker — evidence shows it underperforms on this
  exact regime.
- No multi-object machinery.

This satisfies all stated constraints (inference-only, single-target,
thermal-aware, person-focused) and keeps the architecture modular —
swapping detector implementations only touches one block, exactly as
the assignment's "functional blocks with clear inputs/outputs" framing
asks for.

---

## References

- Suo, J., et al. *HIT-UAV: A high-altitude infrared thermal dataset for
  Unmanned Aerial Vehicle-based object detection.* Scientific Data, 2023.
- Liu, Q., et al. *LSOTB-TIR: A Large-Scale High-Diversity Thermal
  Infrared Object Tracking Benchmark.* ACM MM 2020; IEEE TPAMI 2023.
- Xie, B., et al. *CST Anti-UAV: A Thermal Infrared Benchmark for Tiny
  UAV Tracking in Complex Scenes.* ICCV-W 2025.
- Huang, B., et al. *Anti-UAV410: A Thermal Infrared Benchmark and
  Customized Scheme for Tracking Drones in the Wild.* 2024.
- Devaguptapu, C., et al. *Borrow from Anywhere: Pseudo Multi-modal
  Object Detection in Thermal Imagery (MMTOD).* CVPR-W 2019.
- Yan, B., et al. *LightTrack: Finding Lightweight Neural Networks for
  Object Tracking via One-Shot Architecture Search.* CVPR 2021.
- Liu, S., et al. *Grounding DINO: Marrying DINO with Grounded
  Pre-Training for Open-Set Object Detection.* 2023.
- Bhat, G., et al. *Learning Discriminative Model Prediction for
  Tracking (DiMP).* ICCV 2019.
- Danelljan, M., et al. *Probabilistic Regression for Visual Tracking
  (PrDiMP).* CVPR 2020.
- Wu, X., et al. *Unsupervised Cross-Modal Distillation for Thermal
  Infrared Tracking.* 2021.
- Ye, B., et al. *Joint Feature Learning and Relation Modeling for
  Tracking: A One-Stream Framework (OSTrack).* ECCV 2022.
- Cui, Y., et al. *MixFormer: End-to-End Tracking with Iterative Mixed
  Attention.* CVPR 2022.
