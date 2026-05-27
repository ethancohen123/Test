# Test — Detection & Precision Tracking System

Working repo for Elbit Home Assignment 3. Single-target detection and
precision tracking on aerial thermal footage of a person under
occlusion.

## Repo layout

```
assets/                 input video + assignment PDF
docs/                   working notes (literature review, design,
                        code walkthrough, etc.)
src/baseline/           pipeline modules (see docs/06_code_walkthrough.md)
scripts/                CLI runners
outputs/                generated artefacts (gitignored)
weights/                detector weights (.pt) — gitignored
```

## Setup

```bash
pip install -r requirements.txt   # opencv-contrib-python + numpy + ultralytics
```

The deep-learning pipelines (`dl`, `hybrid`, `follow`) expect a YOLOv8
`.pt` checkpoint at `weights/yolov8_thermal.pt`. The classical
pipelines (`intensity`, `motion`) need no weights.

## Five pipelines

All pipelines run through the same script; the `--pipeline` flag
selects which one. They all write an annotated mp4 to `--out` and
print state-percentage stats at the end. See
[`docs/05_design_review.md`](docs/05_design_review.md) for the
algorithms / math behind each one and
[`docs/06_code_walkthrough.md`](docs/06_code_walkthrough.md) for the
code-level walkthrough.

| Flag | What it is | Uses DL? | Notes |
|---|---|---|---|
| `intensity` | v1 baseline — top-hat morphology + CSRT + Kalman | no | identity-preserving |
| `motion` | unsupervised motion baseline — ego-motion compensated persistence + CSRT + Kalman + AND-gate cross-validation | no | identity-preserving, modality-switch aware |
| `dl` | YOLOv8 + CSRT + Kalman | yes | identity-preserving |
| `hybrid` | DL ∪ motion + CSRT + Kalman + ByteTrack-style two-stage + AND-gate cross-validation | yes | identity-preserving, every block from `motion` plus a person-class prior |
| `follow` | detector-following — DL picks the person each frame, Kalman just smooths + fills gaps | yes | no identity assumption beyond proximity |

### Run any pipeline

```bash
python scripts/run_on_video.py \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --pipeline <name> \
    --out outputs/<name>.mp4 \
    --side-by-side
```

Concrete commands for each:

```bash
# v1 classical baseline (top-hat)
python scripts/run_on_video.py --pipeline intensity \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --out outputs/intensity.mp4 --side-by-side

# Final classical baseline (motion-aware, fully unsupervised)
python scripts/run_on_video.py --pipeline motion \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --out outputs/motion.mp4 --side-by-side

# Pure DL pipeline (thermal-trained YOLOv8 + CSRT + Kalman)
python scripts/run_on_video.py --pipeline dl \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --out outputs/dl.mp4 --side-by-side \
    --dl-weights weights/yolov8_thermal.pt --dl-conf 0.10

# Hybrid (DL + motion + camera-motion-warped Kalman + ByteTrack 2-stage)
python scripts/run_on_video.py --pipeline hybrid \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --out outputs/hybrid.mp4 --side-by-side \
    --dl-weights weights/yolov8_thermal.pt --dl-conf 0.05

# Detector-following (DL decides target per frame, Kalman smooths/fills)
python scripts/run_on_video.py --pipeline follow \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --out outputs/follow.mp4 --side-by-side \
    --dl-weights weights/yolov8_thermal.pt --dl-conf 0.05
```

### CLI options

| Flag | Default | Purpose |
|---|---|---|
| `--video` | (required) | input video path |
| `--out` | `outputs/baseline.mp4` | where to write the annotated mp4 |
| `--pipeline` | `motion` | one of `intensity`, `motion`, `dl`, `hybrid`, `follow` |
| `--side-by-side` | off | render a debug panel beside the main view (persistence heatmap for motion/hybrid/follow, CLAHE gray otherwise) |
| `--max-frames` | 0 | process at most N frames (0 = all) |
| `--init-frame` | 0 | (intensity) frame index at which to seed the track |
| `--init-bbox` | `None` | (intensity) manual init bbox `x,y,w,h` (overrides auto-detection) |
| `--dl-weights` | `weights/yolov8_thermal.pt` | (dl / hybrid / follow) path to the YOLO `.pt` |
| `--dl-conf` | `0.10` | (dl / hybrid / follow) detector confidence floor |
| `--dl-imgsz` | `640` | (dl / hybrid / follow) inference image size |

### Single-frame debug runner

```bash
python scripts/run_on_frame.py \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --frame 120 --out outputs/dbg_frame_120.jpg \
    --side-by-side
```

Useful when you want to look at one specific frame in detail.

### Output legend (same for every pipeline)

The annotated video has a legend in the top-left of the heatmap /
debug panel:

- **Cyan thin box** — per-frame detector candidate (one per blob /
  detection; no identity)
- **Bright-green solid box** — TRACKER, state TRACKING (current
  measurement accepted)
- **Orange dashed box** — TRACKER, state COASTING (Kalman predicting
  blindly, awaiting re-acquisition)
- **Magenta dashed box** — TRACKER, state LOST (gave up; will try to
  re-acquire from any candidate)

## Stats reported at the end

```
processed 1200 frames in 367.08s (3.3 fps)  modality switches: 2
  INIT          0  (  0.0%)
  TRACKING    891  ( 76.6%)
  COASTING    272  ( 23.4%)
  LOST          0  (  0.0%)
```

- `TRACKING %` is **not** an accuracy score — it depends on how
  strictly each pipeline gates "is this really the target?"
- `modality switches` should be 2 on the assignment clip
  (thermal → IR-colormap → thermal).

## Documentation

- [`docs/01_literature_review.md`](docs/01_literature_review.md) —
  classical detection / tracking literature for thermal aerial,
  with the thermal-vs-RGB caveats.
- [`docs/02_active_contours_scan.md`](docs/02_active_contours_scan.md) —
  why active contours / splines were considered and rejected.
- [`docs/03_deep_learning_lit_review.md`](docs/03_deep_learning_lit_review.md) —
  exhaustive DL-for-thermal-SOT survey.
- [`docs/04_hybrid_redesign.md`](docs/04_hybrid_redesign.md) —
  best-in-class hybrid SOT designs (ByteTrack, StrongSORT,
  DeepSORT, …) and what to adopt.
- [`docs/05_design_review.md`](docs/05_design_review.md) — algorithms,
  math, strengths / weaknesses, improvement directions per pipeline.
- [`docs/06_code_walkthrough.md`](docs/06_code_walkthrough.md) —
  beginner-friendly module-by-module code tour.

## Known limitations

- The DL detector (`pitangent-ds/YOLOv8-human-detection-thermal`) is
  white-hot-trained; black-hot frames are bit-inverted before
  inference (see `dl_detector.DLDetector.__call__(invert=...)`).
  Performance still degrades on the rainbow-colormap segment.
- All pipelines except `follow` assume identity preservation; on
  scene-cut-heavy clips this can manifest as the tracker holding
  onto a wrong bbox.
- The HOG re-ID embedding is used because `download.pytorch.org` is
  unreachable from the current sandbox (CNN backend is wired but
  falls back to HOG when offline).
