# Test — Detection & Precision Tracking System

Working repo for Elbit Home Assignment 3. Single-target detection and
precision tracking on aerial thermal footage of a person under
occlusion.

## Layout

```
assets/                 # input video + assignment PDF
docs/                   # working notes (literature review, design)
src/baseline/           # non-DL baseline package
scripts/                # CLI runners
outputs/                # generated artefacts (gitignored)
```

## Setup

```bash
pip install -r requirements.txt   # opencv-contrib-python + numpy
```

## Run

Single annotated frame (debug a particular timestamp):

```bash
python scripts/run_on_frame.py \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --frame 120 --out outputs/dbg_frame_120.jpg --side-by-side
```

Full clip → annotated mp4 (auto-init on the strongest detection):

```bash
python scripts/run_on_video.py \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --out outputs/baseline.mp4 --side-by-side
```

With a manual init bbox (recommended on clips where the strongest blob
is not the target):

```bash
python scripts/run_on_video.py \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --out outputs/baseline.mp4 --side-by-side \
    --init-frame 60 --init-bbox 180,290,30,40
```

## Baseline overview (non-DL)

Pipeline per frame:

1. **Preprocess** — grayscale, polarity detection (white-hot vs
   black-hot, frozen after the first frame), CLAHE, light Gaussian
   denoise.
2. **Detect** — morphological top-hat → adaptive (Otsu + absolute
   floor) threshold → connected components → shape/area gating →
   ranked by local contrast.
3. **Track** — OpenCV CSRT for short-term appearance + Kalman filter
   (constant velocity, state `[cx, cy, w, h]`) for motion prediction.
   When CSRT confidence (NCC vs stored appearance) drops, the pipeline
   falls back to a Kalman-gated re-detection on the current candidate
   list.

States the tracker exposes: `TRACKING` (CSRT measurement accepted),
`COASTING` (Kalman predict only, awaiting re-acquisition), `LOST`
(coast budget exceeded).

See `docs/01_literature_review.md` for the rationale, references, and
the thermal-specific caveats this design tries to handle.

## Known limitations of the baseline

- The clip is a montage with cuts and mixed visualisation modes
  (grayscale thermal, color-mapped thermal). The frozen polarity
  decision is invalid across cuts; `--init-bbox` plus running on a
  single segment is the cleanest demo.
- Without learned appearance, the tracker confuses person-sized warm
  rocks / fauna with the target.
- Long occlusions (> `max_coast_frames`, default 30) end the track.
  These are exactly the failure modes the DL block in step 2 will
  target.
