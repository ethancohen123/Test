# Methods built — short wrap-up

Eight methods were built across the assignment, in two trees that
share I/O but are otherwise independent. Use this doc as the single
"what runs what" reference.

For algorithms / math: [`05_design_review.md`](05_design_review.md).
For code-level walkthrough: [`06_code_walkthrough.md`](06_code_walkthrough.md).
For the v2 freeze in detail: [`08_motion_v2_reference.md`](08_motion_v2_reference.md).

---

## 1. The two trees

```
src/baseline/        post-v2 "principled" tree
                     pipelines: intensity, motion, dl, hybrid, follow, ids
                     runner:    scripts/run_on_video.py --pipeline <name>

src/baseline_v2/     frozen v2 motion baseline (commit 3f3aed1)
                     runner:    scripts/run_motion_v2.py

src/baseline_v2_hybrid/  v2 + DL person-class verifier
                         runner: scripts/run_motion_v2_hybrid.py
```

`baseline_v2/` and `baseline_v2_hybrid/` do not import anything from
`baseline/`; they can be deleted or modified in isolation.

---

## 2. Methods (in build order)

### 2.1 `intensity` — v1 classical baseline (no DL)

Top-hat morphology → adaptive threshold → connected-components →
ranked by local contrast → CSRT + Kalman tracker.

```bash
python scripts/run_on_video.py --pipeline intensity \
    --video assets/...mp4 --out outputs/intensity.mp4 --side-by-side
```

Strength: simple, interpretable, no DL.
Weakness: no motion cue → picks any bright blob (hot rocks, canopy);
auto-init is unreliable.
**Stats on the clip: 66/13/21 % TRACKING/COASTING/LOST.**

### 2.2 `motion` — principled motion baseline (post-v2 tree)

Ego-motion-compensated persistence map + CSRT + camera-motion-warped
Kalman + AND-gate cross-validation (appearance ∧ motion-z) +
Mahalanobis re-acquisition + modality-switch reset.

```bash
python scripts/run_on_video.py --pipeline motion \
    --video assets/...mp4 --out outputs/motion.mp4 --side-by-side
```

Strength: honest tracking; refuses to claim TRACKING without evidence.
Weakness: visually flickery — frequent TRACKING ↔ COASTING flips.
**Stats: 65/21/13 %.**

### 2.3 `dl` — pure YOLO + tracker (no fusion)

Thermal-trained YOLOv8 → CSRT + Kalman + appearance NCC.

```bash
python scripts/run_on_video.py --pipeline dl \
    --video assets/...mp4 --out outputs/dl.mp4 --side-by-side \
    --dl-weights weights/yolov8_thermal.pt --dl-conf 0.10
```

Strength: person-class prior; highest visual TRACKING %.
Weakness: no motion cue when YOLO misses → can drift silently.
**Stats: 83/12/5 %.**

### 2.4 `hybrid` — DL ∪ motion candidates + identity-preserving tracker

Motion-persistence candidates ∪ YOLO candidates, fused by joint
score (DL conf + λ × motion z), then CSRT + camera-motion-warped
Kalman + AND-gate + ByteTrack-style two-stage matching +
Mahalanobis gate + HOG template bank.

```bash
python scripts/run_on_video.py --pipeline hybrid \
    --video assets/...mp4 --out outputs/hybrid.mp4 --side-by-side \
    --dl-weights weights/yolov8_thermal.pt --dl-conf 0.05
```

Strength: all blocks of `motion` plus a person prior; 0 % LOST.
Weakness: motion candidates with high z can outrank legitimate
low-conf DL detections; template bank can be contaminated.
**Stats: 73/26/0 %.**

### 2.5 `follow` — detector-following (philosophy pivot)

YOLO picks the target each frame; Kalman only smooths and fills
short gaps; motion is a fallback when DL is silent.

```bash
python scripts/run_on_video.py --pipeline follow \
    --video assets/...mp4 --out outputs/follow.mp4 --side-by-side \
    --dl-weights weights/yolov8_thermal.pt --dl-conf 0.05
```

Strength: honest — never claims TRACKING without a current
detection on or near the predicted spot.
Weakness: TRACKING % drops to whatever YOLO's per-frame recall is.
**Stats: 42/34/24 %.**

### 2.6 `ids` — multi-object identity tracker + ReID

BoT-SORT-style: list of tracks, three-stage data association
(IoU on high-conf DL → HOG appearance on the rest → re-ID against
a lost-pool of recently-dropped confirmed tracks), DeepSORT-style
two-state lifecycle (tentative → confirmed), camera-motion-warped
Kalman per track, modality-aware reset.

```bash
python scripts/run_on_video.py --pipeline ids \
    --video assets/...mp4 --out outputs/ids.mp4 --side-by-side \
    --dl-weights weights/yolov8_thermal.pt --dl-conf 0.05
```

Strength: persistent integer IDs across frames; same physical
object keeps its ID through short occlusions; lost-pool ReID lets a
track recover its old ID after a multi-second disappearance.
Weakness: multi-box display is busier than single-target; YOLO's
brittle recall means the "right" ID can change after long misses.
**Stats: 48.7 / 51.3 / 0 %, 12 IDs, 10 fps.**

### 2.7 `motion-v2` (frozen) — strongest classical baseline

Snapshot of the motion pipeline at commit `3f3aed1`, **before** the
AND-gate / Mahalanobis / template-bank refinements. Single target,
permissive CSRT acceptance, single-template appearance, fixed-radius
re-acquisition. No DL.

```bash
python scripts/run_motion_v2.py \
    --video assets/...mp4 --out outputs/motion_v2.mp4 --side-by-side
```

Strength: cleanest visual — single bbox, always green, persistence
heatmap on the left; reads as "working" to a non-technical viewer.
Weakness: silent drift — keeps reporting TRACKING when the bbox has
slid onto a similar-looking wrong patch.
**Stats: 85.9 / 12.9 / 1.2 %, ~10 fps.**
Full reference: [`docs/08_motion_v2_reference.md`](08_motion_v2_reference.md).

### 2.8 `motion-v2-hybrid` — v2 + DL person-class verifier (current best)

`motion-v2` runs unchanged. Every K frames a thermal YOLO is
invoked **as a verifier only**:

```
IoU(v2_bbox, YOLO_person) ≥ 0.2  →  DL VERIFIED  (green badge)
YOLO silent                       →  DL silent    (grey badge)
YOLO fires elsewhere, IoU = 0     →  DL CONTRADICTED (red badge)
```

After N consecutive CONTRADICTED ticks, v2's tracker is reseeded
onto the top YOLO detection. v2's motion path and visualisation are
otherwise untouched.

```bash
python scripts/run_motion_v2_hybrid.py \
    --video assets/...mp4 --out outputs/v2_hybrid.mp4 --side-by-side \
    --dl-weights weights/yolov8_thermal.pt --dl-every-k 3
```

Tune knobs:
- `--dl-every-k N` — call YOLO every N frames (1 ≈ 2 fps; 3 ≈ 4 fps;
  10 ≈ 7 fps).
- `--contradicted-before-reseed K` — consecutive contradictions
  before forced reseed (default 3).

Strength: keeps v2's clean single-box visual; uses DL only as a
"is this really a person?" veto; reseeds correct silent-drift cases
that pure v2 cannot recover from.
Weakness: still bound by YOLO's recall — when YOLO is silent for a
long time we trust v2 (which is the right thing for this clip but
fails if v2 is wrong and YOLO is also silent).
**Stats: 93.4 / 6.6 / 0 %, 7 reseed events, 2 modality switches,
~4 fps.**

---

## 3. One-page comparison

| Method | DL? | Identity | TRACKING % | LOST % | Speed | Visual |
|---|---|---|---|---|---|---|
| intensity | no | single | 66 | 21 | ~20 fps | 1 box |
| motion (principled) | no | single | 65 | 13 | ~10 fps | 1 box, often flickery |
| dl | yes | single | 83 | 5 | ~5 fps | 1 box, can drift |
| hybrid | yes | single | 73 | 0 | ~3 fps | 1 box, honest |
| follow | yes | single | 42 | 24 | ~6 fps | 1 box, very honest |
| ids | yes | multi | 48 | 0 | ~10 fps | many boxes, persistent IDs |
| **motion-v2 (frozen)** | no | single | 86 | 1 | ~10 fps | 1 box, cleanest |
| **motion-v2-hybrid** | yes | single | 93 | 0 | ~4 fps | v2 box + DL badge |

`TRACKING %` is partly a permissiveness number, not a pure accuracy
score; the post-v2 methods often have lower TRACKING % because they
refuse to claim a track without cross-validated evidence.

---

## 4. Which to use when

- **Operator demo / clean visual** → `motion-v2-hybrid`. Single
  green box, DL badge confirms / contradicts in the corner.
- **Pure classical reference** → `motion-v2` (no DL dependency).
- **Persistent integer IDs across occlusion** → `ids`.
- **Honest principled reference** → `motion` or `hybrid`.
- **Detector-only** → `dl` or `follow`.

---

## 5. Document map

- [`01_literature_review.md`](01_literature_review.md) — classical
  detection / tracking literature for thermal aerial.
- [`02_active_contours_scan.md`](02_active_contours_scan.md) — why
  splines / active contours were considered and rejected.
- [`03_deep_learning_lit_review.md`](03_deep_learning_lit_review.md)
  — DL-for-thermal-SOT survey.
- [`04_hybrid_redesign.md`](04_hybrid_redesign.md) — best-in-class
  hybrid designs (ByteTrack, StrongSORT, DeepSORT…) and what to
  adopt.
- [`05_design_review.md`](05_design_review.md) — algorithms / math /
  strengths / weaknesses / improvement directions per pipeline.
- [`06_code_walkthrough.md`](06_code_walkthrough.md) — beginner-
  friendly module-by-module code tour.
- [`07_id_tracking_and_reid.md`](07_id_tracking_and_reid.md) —
  multi-object ID + re-ID design and references.
- [`08_motion_v2_reference.md`](08_motion_v2_reference.md) — v2
  freeze reference.
- *this doc* — short "what runs what" summary.
