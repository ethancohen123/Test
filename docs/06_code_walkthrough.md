# Code Walkthrough (beginner-friendly, exhaustive)

This document walks through **every file** in this repository, in
dependency order — from the smallest building blocks at the bottom
to the script you actually run at the top. The goal is that someone
who hasn't worked on the codebase can read this once and understand
how the parts fit, what each function is doing, and why it's
written the way it is.

If the *math* is what you want, read [`05_design_review.md`](05_design_review.md).
If the *code* is what you want, you're in the right place.

---

## 0. The big picture before any code

### 0.1  How the files relate

```
scripts/
  run_on_video.py    ← entry point; user runs this
  run_on_frame.py    ← single-frame debug helper

src/baseline/
  pipeline.py        ← 5 end-to-end pipelines (the "orchestrators")
  tracker.py         ← the Kalman + CSRT + appearance tracker
  reid.py            ← appearance embedding (HOG / CNN)
  dl_detector.py     ← YOLOv8 wrapper (thermal-trained model)
  detector.py        ← classical detectors (intensity + motion blobs)
  motion.py          ← ego-motion estimation + persistence map
  modality.py        ← detects thermal↔IR switches
  preprocess.py      ← polarity detection, CLAHE, denoise
  io_utils.py        ← video reading/writing
  visualize.py       ← bbox / heatmap / legend drawing
  __init__.py        ← marks the folder as a Python package
```

Dependency arrows (who imports whom) — bottom of the list depends
on the things above:

```
io_utils        ─┐
preprocess      ─┤
detector        ─┤
motion          ─┤
modality        ─┼── used by ──►  tracker.py, pipeline.py
reid            ─┤
dl_detector     ─┤
visualize       ─┘
                          tracker.py  ──► used by ──►  pipeline.py
                                                       │
                                                       ▼
                                               scripts/run_on_video.py
```

Read **bottom-up**: start with `io_utils.py`, finish with
`pipeline.py` and the runner. Each file only uses what came before it.

### 0.2  Python idioms used throughout (one-time crash course)

A few patterns repeat. Once you've seen them, the code reads easily.

**Type hints.** When you see something like
`def f(x: int, name: str) -> bool:` the `: int`, `: str`, `-> bool`
are *hints* about expected types. Python doesn't enforce them at
runtime, but tools and humans use them as documentation.

**`np.ndarray`** is the NumPy array type. A grayscale image is a 2-D
ndarray of shape `(H, W)` with `uint8` (0-255) values; a colour
image is a 3-D ndarray of shape `(H, W, 3)` (BGR order in OpenCV).

**`@dataclass`** is a Python decorator that turns a class into a
plain "bundle of named fields" — like a struct in C. Instead of
writing `__init__` by hand, the decorator generates it from the
field declarations:

```python
@dataclass
class Point:
    x: float
    y: float = 0.0      # default value

p = Point(x=3.0)        # auto-generated constructor; p.y == 0.0
```

We use this everywhere for `*Config` objects so the configuration
options are visible at one glance.

**`from __future__ import annotations`** at the top of every file is a
forward-compat hint for type annotations — it lets us write
`tuple[int, int]` even on older Python versions where that syntax
would otherwise error. Safe to ignore mentally.

**`field(default_factory=...)`** in a dataclass means "build a fresh
default value for *each* instance" (avoiding the common pitfall of
sharing a mutable default between instances).

**Lazy import** (`from .reid import X` inside a function rather than
at the top of the file) is used in places where the dependency
(e.g. torch) is heavy or optional. Importing it only when needed
keeps the rest of the code fast and not-broken if the optional
package is missing.

---

## 1. `src/baseline/__init__.py`

```python
"""Non-DL baseline for thermal target detection and tracking."""
```

A single docstring. Its only job is to mark the folder as a Python
"package", so other code can write `from baseline.pipeline import Pipeline`.

---

## 2. `src/baseline/io_utils.py` — Video I/O

### 2.1  What this file does

Three concerns: (a) get **metadata** about a video file, (b) iterate
over its **frames** one at a time, (c) **write** annotated frames
out to a new mp4.

OpenCV exposes a `cv2.VideoCapture` for reading and `cv2.VideoWriter`
for writing — we wrap them in small helpers that are easier to use
and don't leak file handles.

### 2.2  `VideoMeta`

```python
@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    n_frames: int
```

A bundle of four numbers describing a video: its frame size, frame
rate, and how many frames it has. Pure data; no methods.

### 2.3  `probe(path)`

```python
def probe(path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    ...
    meta = VideoMeta(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta
```

Opens the file, asks OpenCV for the four numbers via
`CAP_PROP_*` constants, *releases* the file handle (important — if
you forget, the OS keeps the file locked), returns a `VideoMeta`.
The `or 30.0` is a safety net for files that don't report fps.

### 2.4  `iter_frames(path, start, stop, step)`

```python
def iter_frames(path, start=0, stop=None, step=1):
    """Yield (frame_index, BGR frame) pairs."""
    cap = cv2.VideoCapture(str(path))
    ...
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if stop is not None and idx >= stop:
                return
            if (idx - start) % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()
```

This is a **generator** (the `yield` makes it one). Instead of
loading every frame into a list, it produces one frame at a time as
the caller asks. Used like:

```python
for idx, frame in iter_frames("video.mp4"):
    process(idx, frame)
```

The `try / finally` ensures `cap.release()` runs even if something
breaks in the loop body.

### 2.5  `read_frame(path, index)`

Just seek to a specific frame and read it. Useful for the
single-frame debug script.

### 2.6  `VideoWriter`

```python
class VideoWriter:
    def __init__(self, path, fps, fourcc="mp4v"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._w = None
        self._size = None

    def write(self, frame):
        if self._w is None:
            h, w = frame.shape[:2]
            self._size = (w, h)
            self._w = cv2.VideoWriter(str(self.path), self.fourcc, self.fps, (w, h))
        self._w.write(frame)

    def close(self): ...
    def __enter__(self):  return self
    def __exit__(self, *exc): self.close()
```

Three things to notice:

- **Lazy opening.** OpenCV's writer needs to know the frame size up
  front, but we don't know it until the first frame arrives. So we
  delay creating the writer until `write()` is called for the first
  time, and use that frame's shape.
- **Auto-create parent directory** (`mkdir(parents=True)`) so writing
  to `outputs/baseline.mp4` works even if `outputs/` doesn't exist.
- **Context-manager protocol** (`__enter__` / `__exit__`) so it can
  be used with `with VideoWriter(...) as w:` and the close happens
  automatically. The same pattern Python uses for files.

---

## 3. `src/baseline/preprocess.py` — Thermal-aware grayscale preparation

### 3.1  What this file does

Three jobs: (a) convert BGR to grayscale, (b) decide whether the
scene is **white-hot** (bright = warm) or **black-hot** (dark =
warm), (c) **normalise** with CLAHE so downstream code sees
consistent contrast despite the camera's automatic gain control.

### 3.2  `PreprocConfig`

```python
@dataclass
class PreprocConfig:
    clahe_clip: float = 2.5
    clahe_grid: int = 8
    blur_ksize: int = 3          # 0 disables blur
    assume_white_hot: bool | None = None  # None = auto-detect
```

A bundle of options. The `bool | None` syntax means "either a bool
or None"; here, `None` is the "I don't know yet — figure it out from
the first frame" sentinel.

### 3.3  `to_gray(frame_bgr)`

One-liner that runs `cv2.cvtColor` if the input is 3-channel,
returns the input unchanged if it's already 2-D. Defensive
programming so callers don't need to think about it.

### 3.4  `detect_polarity_white_hot(gray)`

```python
def detect_polarity_white_hot(gray):
    med = np.median(gray)
    upper = float(np.mean(gray > med + 25))
    lower = float(np.mean(gray < med - 25))
    return upper >= lower
```

The intuition: in a real thermal scene, the warm parts are a small
minority of pixels. If the histogram has a **heavier upper tail**
than lower tail (compared to the median), the rare bright pixels
are the warm bodies → white-hot. Otherwise black-hot.

Returns `True` for white-hot. The pipeline decides once per regime
(on the first frame after a modality reset) and freezes the choice.

### 3.5  `apply_clahe(gray, clip, grid)`

```python
def apply_clahe(gray, clip, grid):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)
```

Thin wrapper over OpenCV's CLAHE (Contrast Limited Adaptive Histogram
Equalisation). It divides the image into `grid × grid` tiles, equalises
each tile's histogram, and clips amplification at `clipLimit` to prevent
noise blow-up. Result: locally normalised contrast, robust to the
thermal camera's automatic exposure changes.

### 3.6  `preprocess(frame_bgr, cfg, white_hot)`

```python
def preprocess(frame_bgr, cfg, white_hot):
    gray = to_gray(frame_bgr)
    if not white_hot:
        gray = cv2.bitwise_not(gray)          # invert black-hot → white-hot
    gray = apply_clahe(gray, cfg.clahe_clip, cfg.clahe_grid)
    if cfg.blur_ksize and cfg.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
    return gray
```

The full chain: gray → polarity flip if needed → CLAHE → light
blur. The output is an 8-bit single-channel image where the
**target is always the bright thing**. Every downstream block
assumes this contract.

---

## 4. `src/baseline/detector.py` — Two classical detectors

This file has two detectors and a shared `Detection` dataclass:
**Intensity-based** (top-hat morphology) and **Motion-based**
(persistence-map blobs).

### 4.1  `Detection` dataclass

```python
@dataclass
class Detection:
    bbox: tuple[int, int, int, int]   # (x, y, w, h)
    score: float
    area: int

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0
```

A bounding box plus a confidence-like score plus the area in
pixels. The `@property` decorator turns `center` into an
attribute-like access — `det.center` (no parens), not
`det.center()`.

This is the **common interface** every detector returns. Whether the
detection came from top-hat, motion persistence, or YOLO, it's the
same shape; downstream blocks don't know or care.

### 4.2  Intensity detector (`detect`)

#### 4.2.1  `DetectorConfig`

Knobs for the top-hat detector:

```python
@dataclass
class DetectorConfig:
    tophat_ksize: int = 15       # structuring-element size ≈ target size
    abs_thresh: int = 25         # 0-255 threshold floor on top-hat response
    min_area: int = 8
    max_area: int = 4000
    min_aspect: float = 0.2
    max_aspect: float = 5.0
    open_ksize: int = 3          # post-threshold morph open
    top_k: int = 10              # keep N strongest
```

#### 4.2.2  `_local_contrast(gray, x, y, w, h)`

Compute the LCM ranking score: mean intensity inside the bbox
minus mean intensity in a surrounding ring of the same scale. A
real target should be brighter than its background; this value is
that brightness gap in 8-bit units.

The function is careful to (a) clamp the ring to the image, and
(b) subtract the inner pixels' mass from the outer sum so we don't
double-count.

#### 4.2.3  `detect(gray, cfg)`

```python
def detect(gray, cfg):
    k = max(3, cfg.tophat_ksize | 1)  # force odd
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)

    otsu_thr, _ = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(cfg.abs_thresh, int(otsu_thr))
    _, bw = cv2.threshold(tophat, thr, 255, cv2.THRESH_BINARY)

    if cfg.open_ksize >= 3:
        opk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.open_ksize,)*2)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, opk)

    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    dets = []
    for i in range(1, n_lab):
        x, y, w, h, area = stats[i]
        if area < cfg.min_area or area > cfg.max_area:
            continue
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
        if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
            continue
        score = _local_contrast(gray, x, y, w, h)
        dets.append(Detection(bbox=(int(x), int(y), int(w), int(h)),
                              score=float(score), area=int(area)))
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[:cfg.top_k]
```

Step by step:

1. **Top-hat morphology** (`MORPH_TOPHAT`) extracts bright structures
   smaller than the structuring element.
2. **Threshold**: combine Otsu's automatic threshold with our floor
   so very low-contrast scenes don't generate noise candidates.
3. **Open** (erode then dilate) to kill 1-pixel noise.
4. **Connected components** finds each blob's bounding box and area.
5. **Filter** by area and aspect ratio (a target shouldn't be a
   1×100 sliver).
6. **Score** each by LCM, **sort**, return the top K.

### 4.3  Motion detector (`detect_motion`)

#### 4.3.1  `MotionDetectorConfig`

```python
@dataclass
class MotionDetectorConfig:
    k_mad: float = 4.0          # k-σ above median (anomaly cut)
    abs_floor: float = 2.0      # numerical floor (not video tuned)
    min_area: int = 12
    max_area: int = 6000
    min_aspect: float = 0.15
    max_aspect: float = 6.0
    open_ksize: int = 3
    close_ksize: int = 5
    top_k: int = 10
```

Same kind of knobs, except the threshold is now **data-driven**:
"k × MAD above the median of the persistence map itself", not a
fixed pixel level.

#### 4.3.2  `detect_motion(persistence, cfg)`

```python
def detect_motion(persistence, cfg):
    if persistence is None:
        return []
    p = persistence
    flat = p.ravel()
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med))) * 1.4826  # σ-equivalent
    data_thr = med + cfg.k_mad * mad
    thr = max(cfg.abs_floor, data_thr)
    bw = (p >= thr).astype(np.uint8) * 255
    ...
    # open + close + CC + filter + score
```

Two things worth pointing out:

- **MAD as a robust σ.** `np.std` would be skewed by the few very
  hot pixels of the actual target. The Median Absolute Deviation
  is barely affected by outliers, and multiplying by 1.4826
  reproduces the standard deviation for a Gaussian distribution.
- **Scoring.** Instead of pure sum of persistence (which would
  reward huge warp-residual sheets), the score is
  `mean_persistence × √area` — a real moving target has high mean
  even if it's small; a noise sheet has medium mean but huge area.
  The √ caps the contribution of area.

---

## 5. `src/baseline/motion.py` — Camera-motion compensation

### 5.1  Why this file exists

The drone is moving. So plain frame-differencing is dominated by the
camera's apparent motion, not the target's. The fix: estimate the
inter-frame **homography** (the 3×3 matrix that explains the
camera's motion), warp the previous frame to align with the current
one, and *then* take the difference. Anything still bright in that
difference is moving **against the ground** — that is, against the
warped background.

### 5.2  `MotionConfig`

```python
@dataclass
class MotionConfig:
    orb_n_features: int = 800
    ransac_reproj: float = 3.0
    min_inliers: int = 25
    diff_blur_ksize: int = 5
    persistence_alpha: float = 0.6
    persistence_decay_floor: float = 0.0
```

### 5.3  `estimate_homography(prev_gray, curr_gray, cfg)`

```python
def estimate_homography(prev_gray, curr_gray, cfg):
    orb = cv2.ORB_create(nfeatures=cfg.orb_n_features, fastThreshold=10)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = list(bf.match(des1, des2))
    if len(matches) < cfg.min_inliers:
        return None
    matches.sort(key=lambda m: m.distance)
    matches = matches[: max(80, len(matches) // 2)]

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_reproj)
    if H is None or mask is None or int(mask.sum()) < cfg.min_inliers:
        return None
    return H
```

Three logical phases:

1. **Detect ORB keypoints** in both frames. ORB is a fast, free
   keypoint detector + binary descriptor. Each keypoint is a corner;
   each descriptor is a 256-bit fingerprint.
2. **Match** descriptors by Hamming distance (cheap on binary
   strings). `crossCheck=True` keeps only mutually-best pairs.
   Sort by quality, take the top half (capped at 80).
3. **Estimate the homography H** with RANSAC. RANSAC tries random
   subsets of matches, fits an H, counts how many other matches
   agree within `ransac_reproj` pixels, and keeps the best
   consensus. If too few agree, we return `None` (no usable H).

Returning `None` is a soft failure — the caller falls back to no
motion compensation that frame.

### 5.4  `warp_like(img, H, ref)`

Apply H to project `img` into the coordinate frame of `ref`. Just a
wrapper over `cv2.warpPerspective`.

### 5.5  `compensated_diff(prev_gray, curr_gray, H, cfg)`

```python
def compensated_diff(prev_gray, curr_gray, H, cfg):
    if H is None:
        warped = prev_gray
    else:
        warped = warp_like(prev_gray, H, curr_gray)
    diff = cv2.absdiff(curr_gray, warped)
    if cfg.diff_blur_ksize and cfg.diff_blur_ksize >= 3:
        k = cfg.diff_blur_ksize | 1
        diff = cv2.GaussianBlur(diff, (k, k), 0)
    return diff
```

`cv2.absdiff` is per-pixel `|a - b|`. Blur smooths off
single-pixel sensor noise. Result: a "motion strength" image.

### 5.6  `PersistenceMap`

```python
class PersistenceMap:
    def __init__(self, cfg):
        self.cfg = cfg
        self._map = None

    def reset(self):  self._map = None

    def value(self): return self._map

    def update(self, diff, H):
        d = diff.astype(np.float32)
        if self._map is None:
            self._map = d.copy()
            return self._map
        prev = self._map
        if H is not None and prev.shape == diff.shape:
            prev = warp_like(prev, H, diff)
        a = float(self.cfg.persistence_alpha)
        self._map = a * prev + (1.0 - a) * d
        ...
        return self._map
```

A small class that holds a running average of "what's been moving
lately", but **in the moving frame**: we warp it each step so it
keeps tracking the camera. The EMA formula is

```
P_t = α · warp(P_{t-1}) + (1 - α) · diff_t
```

with α = 0.6 by default → half-life ≈ 1.7 frames. A persistently
moving target stays hot; one-shot warp residuals decay quickly.

---

## 6. `src/baseline/modality.py` — Detecting thermal↔IR switches

### 6.1  Why we care

The uploaded clip has two modality changes (grayscale thermal ↔
rainbow-colormap IR). Across such a switch, polarity, contrast, and
the YOLO model's training distribution all change at once. The
sensible thing is to **reset the temporal state** at each switch.

### 6.2  `ModalityMonitor.step`

```python
def step(self, gray, frame_bgr):
    is_color = self.is_color_frame(frame_bgr)
    hist = self._hist(gray, self.cfg.hist_bins)
    switched = False
    if self._prev_hist is not None and self._cooldown == 0:
        d = self._chi2(hist, self._prev_hist)
        color_flip = is_color != self._last_is_color
        if d > self.cfg.chi2_threshold or color_flip:
            switched = True
            self._cooldown = self.cfg.cooldown_frames
    self._prev_hist = hist
    self._last_is_color = is_color
    if self._cooldown > 0:
        self._cooldown -= 1
    return switched
```

Each frame:

1. Compute a 32-bin normalised intensity histogram.
2. Compare to the **previous** stored histogram with the
   symmetric chi-square distance:
   χ²(a, b) = ½ · Σ (aᵢ − bᵢ)² / (aᵢ + bᵢ + ε)
3. Also check whether the frame is "colour" (`|R - B|` mean > 8).
4. If either the histogram distance is large or the colour flag
   flips, report a **switch** and start a 10-frame cooldown so a
   single regime change can't fire repeatedly.

The pipeline's `step()` method calls `_hard_reset_temporal()` when
this returns True.

---

## 7. `src/baseline/tracker.py` — the Kalman + CSRT + appearance tracker

This is the biggest file. It implements the actual *tracker*
(everything between "we have a candidate detection" and "we have an
identity-preserved bbox per frame").

### 7.1  The shape of a track

```python
class TrackState(Enum):
    INIT = auto()
    TRACKING = auto()
    COASTING = auto()  # Kalman predicting; no measurement
    LOST = auto()      # coast budget exceeded
```

Four-state machine. Every frame the tracker is in exactly one of
these states, and the visualizer colours the bbox accordingly.

```python
@dataclass
class TrackState_:
    bbox: tuple[int, int, int, int]
    state: TrackState
    coast_frames: int = 0
    score: float = 1.0
    appearance: np.ndarray | None = field(default=None, repr=False)
```

The "return type" of `tracker.update()`. Bundles up everything a
caller might need to know about the current frame.

### 7.2  Kalman filter helpers

```python
def _make_kalman() -> cv2.KalmanFilter:
    kf = cv2.KalmanFilter(6, 4)   # state dim 6, meas dim 4
    dt = 1.0
    kf.transitionMatrix = np.array([
        [1, 0, 0, 0, dt, 0],
        [0, 1, 0, 0, 0, dt],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.eye(4, 6, dtype=np.float32)
    kf.processNoiseCov = np.diag([4.0, 4.0, 1.0, 1.0, 9.0, 9.0]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([1.0, 1.0, 4.0, 4.0]).astype(np.float32)
    kf.errorCovPost = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0]).astype(np.float32)
    return kf
```

Concretely:

- **State vector** is 6-dim: `[cx, cy, w, h, vx, vy]`. The first
  four are the bbox; the last two are how fast the bbox is moving.
- **Measurement vector** is 4-dim: `[cx, cy, w, h]` — we measure
  the bbox, not the velocity.
- The transition matrix `F` says "cx_next = cx + vx, w_next = w,
  vx_next = vx" — i.e. constant-velocity for position, constant
  for size and velocity.
- Process noise `Q` is intentionally larger on velocity so that
  velocity uncertainty grows fast during coast.
- Initial covariance `P` has large velocity variance because at
  init we genuinely don't know the velocity (we set vx, vy = 0 but
  with high uncertainty).

```python
def _bbox_to_meas(bbox):  # (x,y,w,h) → [[cx],[cy],[w],[h]]
def _state_to_bbox(state): # [cx,cy,w,h] → (x,y,w,h)
```

Conversion helpers — Kalman uses centre+size, OpenCV uses
top-left+size; we keep both views around and convert at the
boundary.

### 7.3  Appearance helpers

```python
def _appearance_patch(gray, bbox, out_size=32):
    """32×32 mean-subtracted, L2-normalised crop. The legacy
    appearance signal (NCC similarity)."""

def _ncc(a, b):
    if a.shape != b.shape: return -1.0
    return float((a * b).sum())   # dot product of normalised patches
```

NCC = normalised cross-correlation. Two equally-shaped,
mean-subtracted, L2-normalised patches → dot product is
their correlation, in [-1, 1].

```python
def _persistence_z(persistence, bbox):
    """z = (mean persistence inside bbox - global median) / global MAD"""
```

Same z-score logic as the detector, but for any arbitrary bbox.
This is what lets the **AND-gate** ask "does the persistence map
actually show motion at the place we think the target is?".

### 7.4  `TrackerConfig`

```python
@dataclass
class TrackerConfig:
    max_coast_frames: int = 30
    gate_sigma: float = 3.0
    gate_min_radius_px: float = 12.0
    min_appearance: float = 0.30
    min_persistence_z: float = 2.0
    appearance_alpha: float = 0.2
    use_reid: bool = True
    reid_template_bank_size: int = 8
    reid_bank_min_gap: float = 0.985
    priority_min_appearance: float = 0.35

    @property
    def min_appearance_ncc(self) -> float:
        return self.min_appearance
```

The knobs that govern the tracker's behaviour, with the principles
behind each documented in the docstring (see also
[`05_design_review.md`](05_design_review.md)). The `@property` is
a backwards-compat alias.

### 7.5  The `Tracker` class — high-level structure

```python
class Tracker:
    _reid_shared = None   # class-level cache for the ReID extractor

    def __init__(self, cfg=None):
        self.cfg = cfg or TrackerConfig()
        self.kf = _make_kalman()
        self._csrt = None
        self.state = TrackState.INIT
        self.bbox = None
        self.coast_frames = 0
        self.appearance = None
        self.appearance_bank = []
        self.last_score = 0.0
        self._reid = self._get_reid() if self.cfg.use_reid else None
```

Stores:

- A Kalman filter (`kf`)
- An OpenCV CSRT tracker handle (`_csrt`) — re-created on every
  re-acquisition
- A current state + bbox + coast counter
- An appearance template + a **template bank** (the last K accepted
  templates)
- A reference to a shared ReID extractor

The `_reid_shared = None` at *class scope* (not in `__init__`) means
**every Tracker instance shares one ReID extractor** — important
because loading the CNN/HOG is expensive and we don't want to do it
N times when we reset the tracker mid-clip.

### 7.6  `Tracker.init(gray, det, frame_bgr=None)`

Called once when a track is started. Steps:

1. **Sanitise** the bbox (clamp to image, ensure min side 8 px).
   If the bbox is too small or off-frame, give up — the caller
   tries again next frame.
2. **Try CSRT init** with a try/except wrapper, because CSRT can
   raise on some edge-case bboxes.
3. **Reseed Kalman**: copy bbox into Kalman state, **velocity = 0**,
   covariance reset to its initial value.
4. **Take an appearance signature** of the crop (HOG or CNN).
5. **Clear the template bank** and push the first signature.

### 7.7  `Tracker.update(...)`

The main per-frame method. Signature:

```python
def update(self, gray, candidates,
           persistence=None, ego_motion_H=None,
           frame_bgr=None, priority=None) -> TrackState_:
```

What it does in order:

1. **Camera-motion warp**. If a homography is provided, apply it
   to the Kalman state (position and velocity tip) so the predicted
   bbox lives in the current frame's coordinates.
2. **Kalman predict** — advances `statePre` and `errorCovPre`.
3. **Try CSRT** — ask the short-term correlation tracker where it
   thinks the target is.
4. **AND-gate the CSRT measurement.** Compute appearance similarity
   against the template bank and persistence z-score at the CSRT
   bbox. Accept only if both clear their respective thresholds.
   If persistence is `None` (DL pipeline), use appearance alone.
5. **ByteTrack-style priority pool.** If the CSRT measurement was
   rejected and a high-conf priority candidate exists with good
   appearance, accept that one — no spatial gate.
6. **Mahalanobis fallback search** over all candidates. The gate
   uses the Kalman position covariance, so it widens naturally
   during coast.
7. **Update**. On success, Kalman.correct, push appearance to bank,
   reset coast. On failure, advance coast frames; transition
   TRACKING → COASTING → LOST after `max_coast_frames`.

### 7.8  The helper methods (in `Tracker`)

```python
def _reseed_kalman(self, bbox):
    """Snap Kalman state to a measurement, velocity = 0, covariance reset."""

def _warp_state_by_homography(self, H):
    """Apply inter-frame H to position and velocity tip."""

def _make_signature(self, gray, frame_bgr, bbox):
    """Return HOG embedding (if ReID on + we have BGR) else 32×32 patch."""

def _similarity(self, a, b):
    """Cosine if a/b are 1-D embeddings; NCC if 2-D patches."""

def _bank_similarity(self, sig):
    """Best-of-K cosine against the template bank."""

def _push_to_bank(self, sig):
    """Append sig if it's not a near-duplicate of the freshest entry."""

def _sanitise_bbox(bbox, gray, min_side=8):
    """Clip to image bounds and enforce a minimum size; return None if degenerate."""

def _try_csrt_init(gray, bbox):
    """Wrap cv2.TrackerCSRT_create + init in try/except, return None on failure."""

def _mahalanobis_search(self, gray, frame_bgr, candidates, persistence):
    """Pick the best candidate inside the Mahalanobis ellipse (or within
    a spatial floor) that also passes the evidence test (appearance OR
    motion above thresholds)."""
```

The point of having so many small private helpers is **testability +
readability**: each does one thing, and the `update` method reads as
"warp; predict; csrt; gate; priority; fallback; update", which is
exactly the algorithm.

---

## 8. `src/baseline/reid.py` — Appearance embeddings

Two interchangeable feature extractors, both with the same API
(`embed(frame_bgr, bbox) -> np.ndarray`).

### 8.1  `HOGReIDExtractor` (the one we actually use here)

```python
class HOGReIDExtractor:
    def __init__(self, crop_size=64, cell=8, block=2, bins=9):
        win = (crop_size, crop_size)
        block_sz = (block * cell, block * cell)
        block_stride = (cell, cell)
        cell_sz = (cell, cell)
        self.hog = cv2.HOGDescriptor(win, block_sz, block_stride, cell_sz, bins)
        self.crop_size = crop_size

    def embed(self, frame_bgr, bbox):
        x, y, w, h = bbox
        # crop, clip to image, resize to 64×64, convert to gray
        crop = ... (clamp + resize + cvtColor) ...
        feat = self.hog.compute(crop).astype(np.float32).flatten()
        n = float(np.linalg.norm(feat)) + 1e-6
        return feat / n           # L2-normalise so cosine = dot product
```

Histogram of Oriented Gradients (Dalal-Triggs 2005). Each 8×8 cell
gets a 9-bin histogram of gradient orientations; cells are grouped
into 16×16 blocks with stride 8, and each block is L2-normalised
locally. Total length 1764 numbers per 64×64 crop.

### 8.2  `CNNReIDExtractor` (only when offline weights are reachable)

```python
class CNNReIDExtractor:
    def __init__(self, crop_size=64):
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(
            m.features,
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.backbone.eval()
        ...
    def embed(self, frame_bgr, bbox): ...
```

Same interface, but feed the crop through the conv stack of
MobileNetV3-Small (ImageNet weights). Output is a 576-D L2-normalised
vector. Not used in this sandbox because the weights download is
blocked.

### 8.3  `make_reid_extractor()`

```python
def make_reid_extractor():
    try:
        return CNNReIDExtractor()
    except Exception:
        return HOGReIDExtractor()
```

Try CNN first; fall back to HOG if any exception (no network, no
torch, no torchvision …). The `Tracker` calls this once and caches.

---

## 9. `src/baseline/dl_detector.py` — YOLO wrapper

A thin adapter that makes `ultralytics.YOLO` look like every other
detector we have: in, a BGR frame; out, a list of `Detection`s.

### 9.1  `DLDetectorConfig`

```python
@dataclass
class DLDetectorConfig:
    weights: str = "weights/yolov8_thermal.pt"
    conf: float = 0.05
    imgsz: int = 640
    classes: tuple[int, ...] | None = None
    iou: float = 0.45
    max_det: int = 30
    top_k: int = 10
    min_box_side: int = 4
```

The thermal-trained model has a single `HUMAN` class so the
`classes` filter is unused by default; left in for the case where a
larger COCO/HIT-UAV model is plugged in.

### 9.2  `DLDetector.__call__`

```python
def __call__(self, frame_bgr, invert=False):
    import cv2
    if invert:
        frame_bgr = cv2.bitwise_not(frame_bgr)
    kwargs = dict(conf=self.cfg.conf, imgsz=self.cfg.imgsz,
                  iou=self.cfg.iou, max_det=self.cfg.max_det,
                  verbose=False)
    if self.cfg.classes is not None:
        kwargs["classes"] = list(self.cfg.classes)
    results = self.model(frame_bgr, **kwargs)
    ...
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        w, h = x2 - x1, y2 - y1
        if w < self.cfg.min_box_side or h < self.cfg.min_box_side:
            continue
        score = float(box.conf[0])
        dets.append(Detection(...))
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[:self.cfg.top_k]
```

Two interesting bits:

- **`invert=True`** flips the image before inference, used when our
  current polarity is black-hot (the model was trained on white-hot).
- **`__call__`** is Python's "make this instance callable" hook. So
  `detector(frame)` instead of `detector.detect(frame)`. Mirrors the
  functional API of the other detectors.

---

## 10. `src/baseline/visualize.py` — Drawing the output

Just rendering helpers. Worth knowing about because the visual
language (which colour means what) is the user-facing interface.

### 10.1  Colour conventions

```python
CAND_COLOR = (255, 255, 0)        # cyan in BGR  → DETECTOR candidate
TRACK_COLOR = {
    TrackState.TRACKING: (0, 230, 0),     # bright green
    TrackState.COASTING: (0, 165, 255),   # orange (dashed)
    TrackState.LOST:     (220, 0, 220),   # magenta (dashed)
    TrackState.INIT:     (180, 180, 180),
}
HUD_TEXT = (240, 240, 240)
```

Every output frame has a small **legend** drawn in the corner so the
viewer doesn't need to remember the colour code.

### 10.2  The helpers

- `_dashed_rect(img, p1, p2, color, thickness, dash)` — draws a
  dashed rectangle by stepping along each edge and drawing short
  segments. Used for COASTING and LOST so dashed = "we're guessing".
- `draw_candidates(frame, dets)` — thin cyan rectangles for every
  detector candidate (no label).
- `draw_track(frame, track_state, trail)` — bold rectangle in the
  state-colour, with a text label ("TRACK: TRACKING s=0.96"). For
  coasting/lost, dashed. Optional trail of recent positions.
- `draw_hud(frame, idx, n_dets, ts)` — bottom-left text overlay with
  the frame index, candidate count, and current state.
- `draw_legend(frame)` — semi-transparent panel with the colour key.
- `side_by_side(left, right)` — horizontal stack of two images,
  resizing the left to match the right's height. Used to show
  preprocessed gray or persistence-heatmap next to the annotated
  output.
- `persistence_heatmap(pmap, shape)` — render a persistence float
  array as a JET-colormapped BGR image (blue = cold, red = hot).

---

## 11. `src/baseline/pipeline.py` — The five orchestrators

This is where everything we've defined gets wired together into
five end-to-end pipelines. Each one is independent — same input
signature, same output type, different internal flow.

### 11.1  `StepResult` — the per-frame output

```python
@dataclass
class StepResult:
    frame_idx: int
    gray: np.ndarray
    candidates: list[Detection]
    track: TrackState_ | None
    persistence: np.ndarray | None = None
    modality_switched: bool = False
```

Whatever the pipeline did this frame, it returns this. The runner
script uses this to render the annotated frame.

### 11.2  `Pipeline` — v1 intensity-only

```python
class Pipeline:
    def __init__(self, cfg=None):
        self.cfg = cfg or PipelineConfig()
        self._white_hot = self.cfg.preproc.assume_white_hot
        self.tracker = Tracker(self.cfg.tracker)
        self._initialised = False

    def step(self, frame_idx, frame_bgr):
        if self._white_hot is None:
            self._resolve_polarity(to_gray(frame_bgr))
        gray = preprocess(frame_bgr, self.cfg.preproc, self._white_hot)
        cands = detect(gray, self.cfg.detector)
        ts = None
        if not self._initialised:
            if frame_idx >= self.cfg.init_frame_index:
                seed = self._seed_detection(cands)
                if seed is not None:
                    self.tracker.init(gray, seed, frame_bgr=frame_bgr)
                    self._initialised = True
                    ts = TrackState_(...)
        else:
            ts = self.tracker.update(gray, cands, frame_bgr=frame_bgr)
        return StepResult(frame_idx, gray, cands, ts)
```

Linear flow: polarity → preprocess → top-hat detect → init-or-update.

### 11.3  `MotionPipeline` — v2 motion-aware

Adds (over `Pipeline`):

- A `ModalityMonitor` that triggers a hard reset on switches.
- An ego-motion homography per frame.
- A `PersistenceMap` (EMA of compensated diffs).
- A `_is_bad_diff` guard that resets persistence on scene cuts.
- Init based on **persistence z-score streak** (no `init_bbox` arg).
- The tracker.update gets `persistence=pmap` and `ego_motion_H=H`.

The pipeline's `step` is roughly:

```python
def step(self, idx, bgr):
    raw = to_gray(bgr)
    switched = self._modality.step(raw, bgr)
    if switched: self._hard_reset_temporal()
    if self._white_hot is None: ...   # re-detect polarity
    gray = preprocess(bgr, ..., self._white_hot)

    pmap, cands, H = None, [], None
    if self._prev_raw is None:
        self._prev_raw = raw
    else:
        H = estimate_homography(self._prev_raw, raw, self.cfg.motion)
        diff = compensated_diff(self._prev_raw, raw, H, self.cfg.motion)
        self._prev_raw = raw
        if self._is_bad_diff(diff):
            self._persistence.reset()
        else:
            pmap = self._persistence.update(diff, H)
            cands = detect_motion(pmap, self.cfg.detector)

    # init / update logic (with persistence-z streak)
    ...
    return StepResult(...)
```

The `_hard_reset_temporal()` method wipes prev frame, persistence,
polarity, tracker state, streak counter, and `_initialised` — exactly
the set of things that would be invalidated by a modality change.

### 11.4  `DLPipeline` — pure YOLO + tracker

Imports `DLDetector` **lazily** inside `__init__` (so the classical
baselines never need torch).

```python
def step(self, idx, bgr):
    raw = to_gray(bgr)
    if self._white_hot is None: ...
    gray = preprocess(bgr, ..., self._white_hot)
    cands = self.detector(bgr)        # DL detector, BGR input

    if not self._initialised:
        # accept first detection that passes conf threshold and
        # stays in the same neighbourhood for `init_streak` frames
        ...
    else:
        ts = self.tracker.update(gray, cands, persistence=None,
                                  frame_bgr=bgr)
    return StepResult(idx, gray, cands, ts)
```

No persistence map; no ego-motion. The tracker's AND-gate falls back
to appearance-only because we pass `persistence=None`.

### 11.5  `HybridPipeline` — DL ∪ motion with all the bells

This one runs **both** detectors per frame, fuses their outputs, and
feeds the fused list (plus the priority DL detections) to the
tracker. It also propagates the homography for camera-motion warp.

```python
def step(self, idx, bgr):
    # modality + polarity + preprocess (same as MotionPipeline)
    ...

    # Motion path: ego-motion → diff → persistence → motion candidates
    Hmat = None
    pmap = None
    motion_cands = []
    if self._prev_raw is None:
        self._prev_raw = raw
    else:
        Hmat = estimate_homography(self._prev_raw, raw, self.cfg.motion)
        diff = compensated_diff(self._prev_raw, raw, Hmat, self.cfg.motion)
        self._prev_raw = raw
        if self._is_bad_diff(diff):
            self._persistence.reset()
        else:
            pmap = self._persistence.update(diff, Hmat)
            motion_cands = detect_motion(pmap, self.cfg.detector_motion)

    # DL path with polarity correction
    invert = not bool(self._white_hot)
    dl_cands = self.dl(bgr, invert=invert)

    # Fuse: joint score = base + λ * motion_z
    fused = []
    for d in dl_cands:
        z = _persistence_z(pmap, d.bbox) if pmap is not None else 0.0
        joint = float(d.score) + self.cfg.motion_weight * max(z, 0.0)
        fused.append(Detection(d.bbox, joint, d.area))
    for d in motion_cands:
        z = _persistence_z(pmap, d.bbox) if pmap is not None else 0.0
        base = float(np.tanh(max(z, 0.0) / 4.0))
        joint = base + self.cfg.motion_weight * max(z, 0.0)
        fused.append(Detection(d.bbox, joint, d.area))
    fused.sort(key=lambda c: c.score, reverse=True)

    # Init / update (with priority pool for ByteTrack-style)
    if not self._initialised:
        # streak-on-fused-top, same idea as MotionPipeline
        ...
    else:
        priority = [d for d in dl_cands if d.score >= self.cfg.dl_priority_conf]
        ts = self.tracker.update(gray, fused, persistence=pmap,
                                  ego_motion_H=Hmat,
                                  frame_bgr=bgr,
                                  priority=priority)
    return StepResult(idx, gray, fused, ts, persistence=pmap,
                      modality_switched=switched)
```

The two thin things to keep in mind:

- We mix two scoring scales by mapping motion's `tanh(z/4)` to `[0,1)`
  so it can compete with DL conf on roughly the same scale.
- We **don't** dedupe by IoU — both candidates for the same target
  go into the list; the tracker picks. (One of the improvements
  flagged in [`05_design_review.md`](05_design_review.md).)

### 11.6  `FollowingPipeline` — detector-following

A different philosophy. The DL detector picks "the person" each
frame; the Kalman is **only** there to smooth and to coast through
gaps. There is no CSRT, no appearance bank, no identity assumption
beyond "closest detection to the last Kalman prediction".

```python
def _pick_winner(self, dl_cands, motion_cands, pmap,
                  predicted_center, proximity_radius):
    dl_good = [d for d in dl_cands if d.score >= self.cfg.dl_min_conf]

    # Rule 1: closest DL to prediction (proximity gate)
    if predicted_center is not None and dl_good:
        d_closest, det_closest = min((dist_to(d, predicted_center), d)
                                      for d in dl_good)
        if d_closest <= proximity_radius:
            return det_closest

    # Rule 2: strongest DL anywhere (≥ dl_strong_conf)
    strong = [d for d in dl_cands if d.score >= self.cfg.dl_strong_conf]
    if strong:
        return max(strong, key=lambda d: d.score)

    # Rule 3: motion fallback — ONLY when we already have a track
    if predicted_center is not None and motion_cands and pmap is not None:
        ... pick closest motion candidate with z ≥ motion_fallback_z ...

    return None
```

The "only when we already have a track" rule is critical: motion
alone can't tell "is this a person?", so we forbid motion-based
*identity init*.

The `step` method then:

1. Runs modality + preprocess + polarity.
2. Computes ego-motion homography and persistence (for fallback only).
3. Runs the DL detector with polarity-correction.
4. Warps the Kalman state by H (if we have a track).
5. Predicts; gets predicted centre.
6. Picks a winner via `_pick_winner`.
7. If winner: Kalman.correct on it, transition to TRACKING.
8. Else: advance coast; TRACKING → COASTING → LOST.

No CSRT, no appearance bank. Simpler, more directly auditable.

---

## 12. `scripts/run_on_video.py` — the entry point

This is the script `python scripts/run_on_video.py --pipeline X ...`
actually invokes. It:

1. **Parses CLI arguments** (which video, where to write, which
   pipeline, optional DL knobs).
2. **Probes the video** for its metadata.
3. **Builds the pipeline** based on `--pipeline`.
4. **Iterates frames** via `iter_frames`, calls `pipe.step(idx, frame)`
   for each.
5. **Annotates** each frame (candidates + track + HUD), optionally
   prefixed with a side-by-side panel (persistence heatmap for
   motion/hybrid/follow; CLAHE gray for intensity/dl).
6. **Writes** the annotated frame to `outputs/<name>.mp4`.
7. Prints **summary stats** at the end: frame rate, modality switch
   count, and the percentage of frames in each tracker state.

```python
def _build_pipeline(args):
    if args.pipeline == "motion":
        return MotionPipeline(MotionPipelineConfig()), "persistence"
    if args.pipeline == "dl":
        from baseline.dl_detector import DLDetectorConfig
        det_cfg = DLDetectorConfig(weights=args.dl_weights, ...)
        return DLPipeline(DLPipelineConfig(dl_detector=det_cfg)), "gray"
    if args.pipeline == "hybrid": ...
    if args.pipeline == "follow": ...
    # intensity is the default fallback
    cfg = PipelineConfig(init_frame_index=args.init_frame)
    ...
    return Pipeline(cfg), "gray"
```

Returns `(pipeline_instance, side_panel_kind)` — the second value is
just a hint to the renderer about what to put in the left half of
the side-by-side view.

```python
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs/baseline.mp4")
    p.add_argument("--pipeline", choices=[...], default="motion")
    p.add_argument("--side-by-side", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--init-frame", type=int, default=0)
    p.add_argument("--init-bbox", type=str, default=None)
    p.add_argument("--dl-weights", default="weights/yolov8_thermal.pt")
    p.add_argument("--dl-conf", type=float, default=0.10)
    p.add_argument("--dl-imgsz", type=int, default=640)
    args = p.parse_args()
    ...
```

The main loop:

```python
with writer:
    for idx, frame in iter_frames(args.video, stop=stop):
        res = pipe.step(idx, frame)
        if res.modality_switched:
            n_switches += 1
            trail.clear()
        annotated = draw_candidates(frame, res.candidates)
        if res.track is not None:
            x, y, w, h = res.track.bbox
            trail.append((x + w // 2, y + h // 2))
            if len(trail) > 60: trail = trail[-60:]
            annotated = draw_track(annotated, res.track, trail)
            counts[res.track.state] += 1
        annotated = draw_hud(annotated, idx, len(res.candidates), res.track)
        if args.side_by_side:
            left = (persistence_heatmap(res.persistence, frame.shape[:2])
                    if side_kind == "persistence"
                    else res.gray)
            annotated = side_by_side(left, annotated)
        annotated = draw_legend(annotated)
        writer.write(annotated)
```

Everything you've seen in this document is doing its job behind that
one `pipe.step(idx, frame)` call. The runner stays small because
every piece of intelligence lives inside the pipeline.

---

## 13. `scripts/run_on_frame.py` — Single-frame debug helper

Identical pattern, simpler shape: read one frame at `--frame N`,
run one `pipe.step` on it, write the annotated image to disk. Useful
when you want to look closely at a specific moment without rendering
a whole video. Supports `--side-by-side`.

---

## 14. Putting it all together — what happens when you run the script

```bash
python scripts/run_on_video.py \
    --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
    --pipeline hybrid \
    --side-by-side \
    --out outputs/hybrid.mp4
```

For each of the 1200 frames in the input video, the runner calls
`hybrid_pipeline.step(idx, frame)`. Inside that one call:

1. `ModalityMonitor.step` checks for a thermal↔IR switch.
2. `to_gray` + polarity decision + `preprocess` produces CLAHE-gray.
3. `estimate_homography` finds ORB matches and runs RANSAC.
4. `compensated_diff` warps the previous frame and absdiffs.
5. `_is_bad_diff` decides whether to keep this frame.
6. `PersistenceMap.update` blends the diff into the EMA.
7. `detect_motion` thresholds the persistence map and runs CC.
8. `DLDetector(frame, invert=...)` runs YOLOv8 on the (possibly
   bit-inverted) frame.
9. The two candidate lists are **fused** into one list with a joint
   score.
10. If not yet initialised: streak-check the top candidate.
11. If initialised: `Tracker.update` runs camera-motion warp →
    Kalman predict → CSRT update → AND-gate → priority pool →
    Mahalanobis fallback → Kalman correct → bank update.
12. The runner annotates the frame using `draw_candidates`,
    `draw_track`, `draw_hud`, `draw_legend`, `side_by_side`.
13. The annotated frame is written to disk.

At the end, the runner prints state-percentage stats. That's the
entire system, end to end.

---

## 15. Where to start reading the code (suggested order)

If you're picking it up cold:

1. **`io_utils.py`** — small, no dependencies, sets the I/O patterns.
2. **`detector.py`** + **`preprocess.py`** — see what a "detector"
   even is in this codebase.
3. **`motion.py`** — ego-motion is the most novel classical block.
4. **`tracker.py`** — the heart of the system. Skim it once for
   shape, then read it after [`05_design_review.md`](05_design_review.md) §0.7-§0.10.
5. **`pipeline.py`** — read `Pipeline` first (simplest), then
   `MotionPipeline`, then `DLPipeline`, then `HybridPipeline`
   and `FollowingPipeline`. Each builds on the previous one's
   patterns.
6. **`run_on_video.py`** — only ~ 130 lines, mostly argument
   parsing and a render loop.

After that the rest (`reid.py`, `dl_detector.py`, `visualize.py`,
`modality.py`) are small enough to read in one sitting each.

---

## 16. Cheat-sheet of "Wait, where does X happen?"

| You want to find … | It lives in |
|---|---|
| How a video is read frame-by-frame | `io_utils.iter_frames` |
| Polarity decision (white-hot vs black-hot) | `preprocess.detect_polarity_white_hot` |
| The top-hat detector | `detector.detect` |
| The motion-persistence detector | `detector.detect_motion` |
| Camera-motion homography (ORB+RANSAC) | `motion.estimate_homography` |
| The persistence map EMA | `motion.PersistenceMap` |
| The thermal↔IR switch detector | `modality.ModalityMonitor` |
| Kalman filter setup | `tracker._make_kalman` |
| Camera-motion warp of the Kalman state | `tracker.Tracker._warp_state_by_homography` |
| The AND-gate cross-validation | `tracker.Tracker.update`, step 3 |
| ByteTrack-style priority matching | `tracker.Tracker.update`, step 4a |
| Mahalanobis fallback search | `tracker.Tracker._mahalanobis_search` |
| HOG appearance embedding | `reid.HOGReIDExtractor` |
| Template bank | `tracker.Tracker._push_to_bank` / `_bank_similarity` |
| YOLO wrapper | `dl_detector.DLDetector` |
| Polarity-corrected DL inference | `dl_detector.DLDetector.__call__` (`invert=` arg) |
| The colour scheme for the output video | `visualize.TRACK_COLOR` / `CAND_COLOR` |
| The legend in the corner | `visualize.draw_legend` |
| Persistence heatmap rendering | `visualize.persistence_heatmap` |
| Pipeline orchestration | `pipeline.{Pipeline, MotionPipeline, DLPipeline, HybridPipeline, FollowingPipeline}` |
| Pipeline selection from CLI | `scripts/run_on_video.py::_build_pipeline` |
