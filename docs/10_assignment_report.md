# Detection and Precision Tracking System
**Elbit Systems – Video and Analytics Department, Home Assignment 3**

GitHub: this repository.
Demo video: [`assets/motion_v2_hybrid_annotated.mp4`](../assets/motion_v2_hybrid_annotated.mp4)
(single-target box on aerial thermal footage with a DL verifier badge).

---

## 1. System Design

### 1.1 High-level architecture

```
                                ┌─────────────────────┐
                                │ Modality Monitor    │── reset
                                │ (χ² histogram +     │   everything
                                │  colour-flag)       │   on switch
                                └──────────┬──────────┘
                                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Preprocess                                                            │
│ BGR → gray → polarity (white-/black-hot, frozen per regime) → CLAHE  │
│ → light Gaussian blur                                                 │
└────────────────────────────────────┬──────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Ego-motion estimator                                                  │
│ ORB(800) + Hamming BF-match + RANSAC homography H : prev → curr      │
└────────────────────────────────────┬──────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Motion-compensated persistence map                                    │
│ D_t = |I_t − warp(I_{t-1}, H)|                                       │
│ P_t = α·warp(P_{t-1}, H) + (1−α)·D_t      (α = 0.6)                  │
│ + "bad-diff" guard (scene cut → reset P)                             │
└────────────────────────────────────┬──────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Motion detector                                                       │
│ adaptive threshold (abs floor + 99th-pct) → open + close → connected │
│ components → area/aspect filter → score = mean(P)·√area              │
└────────────────────────────────────┬──────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Identity-preserving tracker (CSRT + Kalman)                          │
│ Kalman 6-D constant-velocity; CSRT short-term appearance; appearance │
│ NCC patch (32 × 32 mean-subtracted) gate ≥ 0.3 on CSRT measurements; │
│ re-acquisition in a fixed Mahalanobis-like gate; init from a 3-frame │
│ motion-z streak                                                      │
└────────────────────────────────────┬──────────────────────────────────┘
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ★ DL verifier (every K frames)                              **DL block** │
│ YOLOv8n-thermal ('HUMAN' class) → for each detection compute         │
│ IoU(det, current_track_bbox). Status:                                │
│   max-IoU ≥ 0.2          → VERIFIED                                  │
│   no detection at all    → SILENT  (trust tracker)                   │
│   detections elsewhere,                                              │
│   max-IoU = 0            → CONTRADICTED                              │
│ After N consecutive CONTRADICTED ticks → reseed tracker from top DL  │
│ detection.                                                           │
└────────────────────────────────────┬──────────────────────────────────┘
                                     ▼
                              annotated frame
                              (bbox + DL badge)
```

### 1.2 Per-block specification

| # | Block | Input | Output | Algorithm | Key assumptions | Trade-offs | Risks / limitations |
|---|---|---|---|---|---|---|---|
| 1 | Modality Monitor | raw gray + BGR | switch event (bool) | χ² distance between consecutive 32-bin grayscale histograms + colour-flag flip; 10-frame cooldown | Within a regime, scene statistics are roughly stationary | False positives on fast pans; missed switches inside slow fades | Resets every downstream block — false trigger costs ~10 init frames |
| 2 | Preprocess | BGR frame | normalised 8-bit gray | polarity-detect, optional invert, CLAHE (clip 2.5, 8×8), Gaussian blur | Polarity is fixed within a regime | CLAHE itself induces drift on textureless areas | Wrong polarity → motion path inverts target/background |
| 3 | Ego-motion | (prev, curr) gray | homography H or `None` | ORB (800 feat.) + Hamming BF match (top half, ≤ 80 pairs) + RANSAC (3-px tol, ≥ 25 inliers) | Scene is roughly planar at long range; ≥ 25 stable corners exist | Fails on textureless sky / canopy | Without H, motion path degrades to plain frame differencing |
| 4 | Persistence map | (D_t, H) | float map P_t | warp prior P by H; EMA blend (α = 0.6); bad-diff guard resets on χ² anomaly | Target's apparent motion ≫ residual after warp | α too large → trailing ghost; too small → flicker | Long-stationary target decays to noise floor in ~3 frames |
| 5 | Motion detector | P_t | list of bboxes + scores | threshold = max(abs_floor, 99th-pct); morph open + close; CC + area/aspect gate; score = mean(P) · √area | Target appears as a coherent blob, not a dispersed cloud | Hot rocks / canopy gaps look identical to a moving person | No person prior — scores anything that moves |
| 6 | Tracker | gray + candidates | single bbox + state | Kalman 6-D const-vel; CSRT short-term; NCC patch appearance EMA; init streak (3 frames, z ≥ 4) | Identity is unique and continuous within a regime | Permissive NCC gate → silent drift onto similar patches | One tracked target only; no multi-person |
| 7 | **DL verifier** | BGR frame + tracker bbox (every K frames) | status + optional reseed | YOLOv8n-thermal `HUMAN` class, conf ≥ 0.1; max IoU with tracker bbox classifies VERIFIED / SILENT / CONTRADICTED; N consecutive CONTRADICTED → reseed | YOLO sometimes fires on the target (recall ≥ 1 frame per second is enough) | Higher K → cheaper inference but slower drift detection | YOLO blind spots (heavy occlusion, colormap segment) → DL goes SILENT |
| 8 | Visualiser | annotated frame | mp4 frame | bbox in TRACKING / COASTING / LOST colour + DL-status badge + HUD | – | – | – |

System budget: ~4 fps on a single CPU thread (YOLO is the bottleneck at K = 3); ~10 fps if DL is disabled.

---

## 2. Deep Learning Analysis

The DL block is **block 7 (DL verifier)** — a thermal-aerial person
detector used in a verify-and-correct loop on top of a classical
motion+CSRT tracker.

### 2.1 Literature review — three approaches considered

| # | Family | Representative work | Why it fits our problem |
|---|---|---|---|
| **A** | YOLOv8-thermal | *pitangent-ds/YOLOv8-human-detection-thermal* (HuggingFace); base architecture from Jocher et al. 2023 | Single-shot detector with a native `HUMAN` class trained on thermal images. Small (≈ 3 M params), CPU-feasible, returns confident bboxes that drop straight into our IoU gate. |
| **B** | Siamese / Transformer SOT | SiamFC (Bertinetto 2016), OSTrack (Ye 2022), MixFormer (Cui 2022) | Frame-to-frame template matching. Conceptually elegant; in practice the literature (CST Anti-UAV, ICCV-W 2025) shows transformer SOTs collapse on tiny TIR targets with camera motion — exactly our regime. |
| **C** | Online discriminative tracker | DiMP / PrDiMP (Danelljan 2019/2020), KeepTrack (Mayer 2021) | Best raw TIR-SOT numbers in the literature (LSOTB-TIR benchmark). Requires online gradient updates → not feasible CPU-only / inference-only. |

### 2.2 Model selected

**A: YOLOv8n-thermal** (`pitangent-ds/YOLOv8-human-detection-thermal`),
≈ 3 M params, single class `HUMAN`, ImageNet-style 640-px inference.

### 2.3 Justification

- **Person-class prior** the classical motion path fundamentally lacks
  (motion alone cannot answer "is this thing actually shaped like a
  person?").
- **Pretrained on thermal** — minimal domain gap vs an RGB-only
  COCO YOLO (we empirically measured: COCO yolov8n on our clip finds
  the person in 2/9 sampled frames; the thermal model finds them in
  4/9 at higher confidence).
- **CPU-feasible** — ≈ 150 ms / frame at 640 px on a single thread.
  Sub-sampling to once every K = 3 frames keeps the system close to
  realtime.
- **Bounding-box output is the right granularity** for an IoU-based
  verification gate. Mask / heatmap models would over-specify.
- **Open weights, no fine-tuning required** — matches the
  inference-only constraint of the assignment.

### 2.4 Pros, cons, limitations of the selected model

**Pros**
- Native thermal training distribution.
- Single class → no class-confusion overhead, ~3 M params.
- Drops behind a confidence gate cleanly.

**Cons / limitations**
- Training data is **street-level**, not aerial top-down. Recall on
  small-scale, top-view targets is materially lower than the model's
  benchmark numbers suggest.
- **No identity** — gives bboxes per frame; cannot say "this is the
  same person we saw two seconds ago." That's why we keep the
  classical CSRT + Kalman as the identity layer.
- **Polarity-sensitive** — we verified empirically that bit-inverting
  the input *destroys* recall on this checkpoint (it was apparently
  trained on both polarities; inversion adds artefacts at borders).
  We pass raw frames.
- **Modality-blind** — collapses on the rainbow-colormap IR segment
  the clip contains, going SILENT for ~50 % of those frames.

### 2.5 What we explicitly did NOT do (and why)

- **No fine-tuning** — out of scope (inference-only).
- **No Grounding DINO / OWL-ViT** — open-vocabulary detection sounds
  attractive but blocked by the sandbox (weights live on HF).
- **No DiMP / transformer SOT** — literature evidence (CST Anti-UAV
  ICCV-W 2025: best transformer SOT achieves 35.9 % state accuracy on
  tiny TIR targets vs 67.7 % on regular ones) — they degrade in
  exactly our failure regime.

---

## 3. Success Criteria

### 3.1 Per-block metrics

| Block | Metric | Target on our clip |
|---|---|---|
| Modality Monitor | switch precision / recall against ground-truth segment boundaries | precision ≥ 0.9, recall ≥ 0.9 (clip has 2 GT switches; we detect 2) |
| Preprocess | polarity-correct rate per regime | ≥ 0.95 |
| Ego-motion | inlier ratio of RANSAC homography | ≥ 0.5 of matches; ≥ 25 absolute inliers |
| Persistence map | foreground SNR = mean(P inside GT bbox) / median(P) | ≥ 3 on visible-target frames |
| Motion detector | per-frame recall on GT (IoU ≥ 0.3); precision @ top-K = 10 | recall ≥ 0.8, precision ≥ 0.4 on visible frames |
| Tracker | Tracking-state share (TRACKING / total) | ≥ 0.85 with v2 settings |
| DL verifier | per-tick fraction VERIFIED conditional on YOLO firing | ≥ 0.6 in good segments; SILENT in bad ones is acceptable |
| Visualiser | latency added per frame | ≤ 5 ms |

### 3.2 End-to-end metrics (MOT-style, single-target)

| Metric | Definition | Target |
|---|---|---|
| **Visible-target track rate** | fraction of GT-visible frames in which the predicted bbox has IoU ≥ 0.3 with GT | ≥ 0.7 |
| **Centre-error (px)** | mean Euclidean distance, predicted centre vs GT centre, on TRACKING frames | ≤ 30 px on a 360 × 640 frame |
| **Mean coast duration** | average frames per COASTING run | ≤ 30 (≈ 1 s @ 30 fps) |
| **ID switches** | number of times the *identity* changes hands during the clip | ≤ 1 (single-target system) |
| **Wall-clock fps (CPU)** | end-to-end | ≥ 4 fps with DL block on; ≥ 10 fps without |

Note: we did not hand-label ground truth on this clip, so the numbers
above are *targets*. The reported runtime stats
(85.9 / 12.9 / 1.2 % for v2; 93.4 / 6.6 / 0 % for v2-hybrid) are
internal state shares, not accuracy scores.

---

## 4. Failure Analysis

| # | Failure mode | Root cause | Where it manifests | Impact | Already mitigated? |
|---|---|---|---|---|---|
| F1 | **Silent drift onto a hot rock / similar bright patch** | NCC patch appearance is too permissive; the EMA absorbs the new patch and keeps validating | v2 motion baseline, mid-clip | Tracker reports TRACKING on wrong object indefinitely | DL verifier (block 7) detects CONTRADICTED and reseeds |
| F2 | **Tracker box drifts off-screen during long coast** | Constant-velocity Kalman + permissive bbox clipping → 4-px sliver at image edge stays visible | Any pipeline with a long coast budget; was visible in early ID-pipeline output | Confusing display; can also catch a wrong reseed | Off-screen Kalman-centre guard moves the track to LOST |
| F3 | **Person is hidden under canopy / lies still** | Persistence map decays in ~3 frames; YOLO can't see them either | Mid-clip when target hides | Tracker enters COASTING then LOST | Inherent to the problem; only fix is a wider sensor / second view |
| F4 | **Modality switch (thermal → rainbow IR)** | DL model trained on grayscale; persistence map polarity flips; CLAHE adapts wrong | Around frames 957–985 and 1030–1038 in the clip | All downstream blocks invalidated for ~30 frames | Modality Monitor (block 1) hard-resets the temporal state at the switch |
| F5 | **Scene cuts (montage edits)** | Inter-frame homography fails; diff is essentially "everything moved" | Throughout the assignment clip (uploaded video is a montage) | Persistence map blows up; tracker may follow camera shake | Bad-diff guard zeros out P_t and refuses to update the tracker on that frame |
| F6 | **YOLO false positive on canopy / NUC artefact** | Thermal YOLO mis-classifies bright vegetation patches as "HUMAN" with low conf | Throughout the clip | A spurious CONTRADICTION can trigger a wrong reseed | N consecutive CONTRADICTED required (default N = 3) — one-frame false positives are absorbed |
| F7 | **YOLO blind in colormap segment** | Pretrained on grayscale thermal, not rainbow IR | Frames 970–1030 in the clip | DL verifier goes SILENT; system reverts to pure-v2 behaviour there | Acceptable — v2 motion path still operates, just without verification |
| F8 | **Two visually-similar people** | HOG / NCC appearance bank cannot distinguish them | Not present in this clip; would matter in real deployment | ID switch on cross-over | Would need a CNN ReID embedding — sandbox blocks the weights |

The dominant failure mode the system **does** mitigate is **F1**
(silent drift); this is what the v2 → v2-hybrid upgrade buys.
**F3 and F8** are not mitigated and are the principled limits of the
current design.

---

## 5. Improvement Suggestions

In rough order of expected impact for *this* deliverable:

1. **A better thermal-aerial detector.** The single biggest gain
   would come from a YOLO (or DETR) fine-tuned on HIT-UAV or
   FLIR-ADAS-aerial. Every downstream block inherits the detector's
   per-frame recall as an upper bound on honest TRACKING %. Our
   pipeline is detector-agnostic; the swap is a one-line config
   change.

2. **CNN ReID embedding to replace the patch / HOG appearance**.
   MobileNetV3-Small or a lightweight ReID head (e.g. OSNet) would
   give cosine-distance matching that survives small viewpoint /
   scale changes, fixing the silent-drift class of bugs more
   robustly than the IoU-veto we have now. The infrastructure is
   already wired (`reid.CNNReIDExtractor`); only the weights need
   to be reachable.

3. **Long-term ReID lost-pool**, BoT-SORT / DeepSORT style: when the
   target disappears for many seconds and reappears, a 100-frame
   appearance bank can re-issue the same identity. Already prototyped
   in our `--pipeline ids` mode; should be combined with the v2
   verifier to give the best of both worlds.

4. **Optical-flow-based motion model for the Kalman**. Instead of a
   plain constant-velocity prior, propagate the bbox along the local
   median optical-flow vector at the target — much better short-term
   prediction during coast.

5. **Multi-detector ensemble.** Run both a thermal YOLO and a (small)
   open-vocab detector (Grounding DINO ≥ 1.5 nano variant if a
   release becomes accessible) and vote. Reduces the failure mode of
   a single detector's blind spot.

6. **A small classifier in the verifier** that knows the *current*
   target's appearance and rejects DL detections that look like
   distractors (KeepTrack-style "track what not to track" model).
   Cleaner than a hard reseed and survives close-by similar-looking
   bodies.

7. **Gaussian-smoothed gap interpolation** (StrongSORT GSI): when a
   track recovers after a coast period, retro-actively smooth the
   bbox through the gap rather than rendering the Kalman zig-zag.
   Cosmetic but materially improves the final demo video's
   readability.

---

## Appendix — code & demos

| Artefact | Path |
|---|---|
| Source code | this repository (`src/baseline_v2/`, `src/baseline_v2_hybrid/`, `src/baseline/` for ablations) |
| **v2 baseline (no DL)** runner | `scripts/run_motion_v2.py` |
| **v2 + DL verifier** runner (this report's system) | `scripts/run_motion_v2_hybrid.py` |
| Demo video, v2 only | [`assets/motion_v2_annotated.mp4`](../assets/motion_v2_annotated.mp4) |
| Demo video, v2 + DL verifier | [`assets/motion_v2_hybrid_annotated.mp4`](../assets/motion_v2_hybrid_annotated.mp4) |
| Algorithm + math reference | [`docs/05_design_review.md`](05_design_review.md) |
| Module-by-module code walkthrough | [`docs/06_code_walkthrough.md`](06_code_walkthrough.md) |
| Methods summary | [`docs/09_methods_summary.md`](09_methods_summary.md) |

**Run the system end-to-end:**

```bash
pip install -r requirements.txt
python scripts/run_motion_v2_hybrid.py \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --out outputs/system.mp4 \
    --dl-weights weights/yolov8_thermal.pt \
    --dl-every-k 3
```
