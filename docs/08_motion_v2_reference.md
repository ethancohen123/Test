# Motion-v2 Baseline — Reference

> *The strongest purely-classical (no DL) tracker built during this
> project, frozen as a standalone module so it can be re-run, compared
> against, or wrapped by a hybrid without touching the rest of the
> codebase.*

## TL;DR

```bash
python scripts/run_motion_v2.py \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --out outputs/motion_v2.mp4 \
    --side-by-side
```

- One green/orange/magenta box per frame (single-target).
- One JET-coloured "persistence heatmap" panel on the left when
  `--side-by-side` is on — that's the temporal motion evidence the
  tracker is using.
- ~10 fps CPU, no DL weights required, no torch needed.
- Stats on the assignment clip: **TRACKING 85.9 % / COASTING 12.9 %
  / LOST 1.2 %**, 2 modality switches.

The code lives entirely under `src/baseline_v2/` and the entry point
is `scripts/run_motion_v2.py`. It does not import from
`src/baseline/` (the post-v2, "principled" tree), so it can be
deleted, modified or extended in isolation.

---

## 1. What problem v2 solves

A drone is flying over outdoor terrain; a person is walking inside the
field of view; the camera is **moving** and the **target is moving**.
We want a single bounding box that follows the person for as long as
possible without any manual initialisation, with no learned models,
on CPU, in real-ish time.

That is *the* canonical aerial-thermal SOT problem, and the v2 design
is built around two observations:

1. **The single strongest classical cue when the camera moves is
   motion against the warped background**, i.e. what's moving *relative
   to the ground* after we cancel the drone's apparent motion.
2. **A short-term correlation tracker (CSRT) is enough** for
   inter-frame continuity, as long as we keep feeding it fresh
   measurements from the motion path when it loses lock.

---

## 2. Per-frame algorithm (block-by-block)

```
                        ┌───────────────────────┐
                        │ ModalityMonitor       │ ─► on switch:
                        │ (χ² histogram + colour│    hard-reset
                        │  flag)                │    everything
                        └───────────┬───────────┘
                                    ▼
                  ┌─────────────────────────────────────┐
                  │ to_gray + polarity (white-/black-   │
                  │ hot) + CLAHE + light Gaussian blur  │
                  └─────────────────┬───────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │ ORB + RANSAC homography H : prev_raw → curr_raw   │
        │ (raw grayscale; CLAHE would itself drift)         │
        └───────────────────────────┬───────────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │ Compensated diff   D_t = |I_t − warp(I_{t-1},H)| │
        │ then Gaussian-blur                                │
        └───────────────────────────┬───────────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │ Bad-diff guard (heuristic):                        │
        │ if median(D_t)>30 or  fraction(D_t>50)>0.30        │
        │ → wipe persistence, skip this frame                │
        └───────────────────────────┬───────────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │ Persistence-map EMA:                               │
        │ P_t = α · warp(P_{t-1}, H) + (1−α) · D_t           │
        │ α = 0.6 (half-life ≈ 1.7 frames)                   │
        └───────────────────────────┬───────────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │ Detect on P_t:                                     │
        │   • threshold = max(abs_thresh, percentile-99)     │
        │   • morphological open + close                     │
        │   • connected components + area/aspect filter      │
        │   • score = mean(P) · √area    (top-K = 10)        │
        └───────────────────────────┬───────────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │ Init (once): top candidate's persistence-z ≥ 4    │
        │ for K = 3 consecutive frames within frame-diagonal│
        │ × 12 % → seed CSRT + Kalman                       │
        └───────────────────────────┬───────────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────┐
        │ Per-frame update:                                  │
        │ 1. Kalman.predict                                  │
        │ 2. CSRT.update → bbox                              │
        │ 3. NCC(patch, stored_patch) ≥ 0.30 → accept       │
        │    else search candidates in fixed Mahalanobis-    │
        │    like gate (radius = 3·max(w,h))                 │
        │ 4. Kalman.correct, EMA-update appearance patch     │
        │ 5. Counter coast_frames; LOST after 30             │
        └───────────────────────────┬───────────────────────┘
                                    ▼
                              annotated frame
```

Every block is small (≤ ~100 LOC) and lives in its own file under
`src/baseline_v2/`.

---

## 3. The math, condensed

### 3.1 Pre-processing

Polarity decision on first frame of a regime:
$$
u = \mathbb{P}(I > \mathrm{med}(I) + 25), \qquad
l = \mathbb{P}(I < \mathrm{med}(I) - 25), \qquad
\text{white-hot} \;\iff\; u \ge l.
$$
If black-hot, replace `I ← 255 − I` so the target is always bright
downstream.

CLAHE: `I_clahe = CLAHE(I; clip=2.5, tile=8×8)`.

### 3.2 Ego-motion

ORB (800 features) → Hamming brute-force match with cross-check →
RANSAC homography, reprojection tolerance 3 px, ≥ 25 inliers required.

$$ \tilde{p}_t \;\simeq\; H \, \tilde{p}_{t-1} $$
in homogeneous coordinates. `H = ∅` if RANSAC fails (we silently skip
motion compensation that frame).

### 3.3 Compensated diff + persistence

$$
\hat{I}_{t-1} = \mathrm{warp}(I_{t-1}, H), \qquad
D_t = G_{5\times5} \;*\; |I_t - \hat{I}_{t-1}|,
$$
$$
P_t \;=\; \alpha \cdot \mathrm{warp}(P_{t-1}, H) \;+\; (1-\alpha)\cdot D_t,
\qquad \alpha = 0.6.
$$

### 3.4 Detector on `P_t`

Threshold combines an absolute floor with a 99th-percentile cut on
the map itself (very permissive — *not* the MAD-based threshold that
came in the later "principled" rewrite).

For each connected component:
$$
s_{\text{motion}} \;=\; \overline{P_t}\bigr|_{\text{bbox}} \,\cdot\,
\sqrt{\text{area}}.
$$

### 3.5 Tracker

State vector $x = [c_x,\,c_y,\,w,\,h,\,v_x,\,v_y]^\top$,
measurement $z = [c_x,\,c_y,\,w,\,h]^\top$.

```
F = [I_4  dt·B;   measurement = first 4 rows of state
     0    I_2]
Q = 1e-2 · diag(1, 1, 1, 1, 4, 4)            # very small noise
R = 1.0  · I_4
P_post(init) = I_6
```

Appearance signature: a 32×32 crop, mean-subtracted and L2-normalised.
Similarity = NCC = dot product.

Acceptance rule (this is the *key* difference vs the principled later
versions): the CSRT measurement is accepted if `NCC ≥ 0.30`. **No
persistence-z cross-check, no template bank, no Mahalanobis ellipse.**
If the NCC check fails, the search-in-gate fallback picks the best
appearance match within `redetect_gate_scale × max(w,h) = 3 · max(w,h)`
pixels of the predicted centre.

### 3.6 Modality monitor

Per-frame 32-bin normalised intensity histogram; symmetric χ² distance
to the previous accepted histogram, plus a "is this frame colour?"
flag (`mean |R − B| > 8`). On a switch (χ² > 1.5 or colour-flag flip),
**hard-reset** the persistence map, polarity, init streak, and tracker.

---

## 4. Why v2 reads as "working" visually

Three structural reasons, all amounting to "lenient and consistent":

1. **Single-target, single bbox.** Nothing to disambiguate — one box,
   one identity, one trail.
2. **Permissive CSRT acceptance.** As long as the CSRT output looks
   like the (drifting) appearance EMA, the box stays green. The
   appearance gradually adapts to whatever the box sits on, so the
   match keeps passing the test even if the box has slid onto a
   similar-looking but wrong patch.
3. **No "I'm honestly not seeing it" mechanism.** v2 reports
   `TRACKING 85.9 %` — that is *not* an accuracy number, it's a
   permissiveness number. The later AND-gate / Mahalanobis-gate /
   two-state additions deliberately reduce TRACKING% by refusing to
   claim a track without cross-validated evidence.

For a human watching the output, the v2 reading "always a green box
on something moving" looks like success; later versions' more honest
"sometimes COASTING / LOST when the cross-evidence isn't there" looks
like failure. That trade-off is real and it is the explicit reason
v2 was promoted to a permanent reference.

---

## 5. What v2 does badly

- **Silent drift.** If the appearance EMA slides onto a near-target
  patch (a hot rock that resembles the person), v2 will report
  TRACKING on the wrong object indefinitely. There's no second
  evidence source pushing back.
- **Single fixed-radius re-acquisition gate.** Doesn't widen with
  Kalman uncertainty during coast → if the target reappears at a
  different location than predicted, the search fails.
- **No identity beyond the current track.** After LOST, a new init
  produces "the same identity" only by coincidence.
- **Modality reset is hard.** Anything continuing across a thermal↔IR
  switch is forgotten.

These are exactly the failure modes the post-v2 work was trying to
address; the trade is "less drift" vs "more flicker", and v2 sits
firmly on the "less flicker" side.

---

## 6. Code map (mirrors `src/baseline/` of the same era)

```
src/baseline_v2/
  __init__.py
  io_utils.py        VideoCapture/VideoWriter wrappers, frame iterator
  preprocess.py      polarity detection, CLAHE, blur
  modality.py        χ² histogram switch detector + cooldown
  motion.py          ORB+RANSAC homography, compensated diff, PersistenceMap
  detector.py        intensity (top-hat) + motion (persistence) detectors
  tracker.py         single-target CSRT + Kalman + simple NCC patch ReID
  pipeline.py        Pipeline (intensity) and MotionPipeline (v2 default)
  visualize.py       bbox / heatmap / trail / HUD drawing
scripts/
  run_motion_v2.py   standalone entry point (does not import from
                     src/baseline/, only src/baseline_v2/)
```

`scripts/run_motion_v2.py --pipeline motion` is the v2 motion
baseline; `--pipeline intensity` is v1 (top-hat). The CLI also
supports `--max-frames`, `--init-frame`, `--init-bbox` (same options
as the original).

---

## 7. When to use v2 instead of the post-v2 pipelines

- As a **clean, reproducible classical baseline** for any comparison
  table — its numbers are deterministic and there are no DL weights
  involved.
- As the **input to a hybrid** that adds a DL detector / DL ReID
  layer on top — the v2 motion path is the "what's moving" signal
  that the hybrid can cross-check against the DL "is this a person?"
  signal.
- When **visual clarity matters more than honesty** about uncertain
  frames (e.g., a demo to a non-technical audience that watches the
  bbox as the deliverable).

Use the post-v2 pipelines (`motion`, `hybrid`, `follow`, `ids` in
`src/baseline/`) when **principled-honesty matters more than visual
continuity**.
