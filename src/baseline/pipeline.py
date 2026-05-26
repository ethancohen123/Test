"""End-to-end baseline pipelines.

`Pipeline` — v1 intensity-only baseline (top-hat + CSRT).
`MotionPipeline` — v2 unsupervised motion-aware baseline: ego-motion
    compensated temporal persistence drives both detection and auto-init.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .detector import (Detection, DetectorConfig, MotionDetectorConfig, detect,
                        detect_motion)
from .modality import ModalityConfig, ModalityMonitor
from .motion import MotionConfig, PersistenceMap, compensated_diff, estimate_homography
from .preprocess import (PreprocConfig, detect_polarity_white_hot,
                          preprocess, to_gray)
from .tracker import Tracker, TrackerConfig, TrackState, TrackState_


@dataclass
class PipelineConfig:
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    init_frame_index: int = 0   # frame on which we initialise the track
    init_bbox: tuple[int, int, int, int] | None = None
    # If set, force this bbox at init_frame_index instead of using the
    # top-ranked detection. Lets the operator point at the real target on
    # clips where the strongest blob is not the person.


@dataclass
class StepResult:
    frame_idx: int
    gray: np.ndarray             # preprocessed bright-target image
    candidates: list[Detection]
    track: TrackState_ | None    # None until init succeeds
    persistence: np.ndarray | None = None   # MotionPipeline only
    modality_switched: bool = False


class Pipeline:
    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg or PipelineConfig()
        self._white_hot: bool | None = self.cfg.preproc.assume_white_hot
        self.tracker = Tracker(self.cfg.tracker)
        self._initialised = False

    def _resolve_polarity(self, first_gray: np.ndarray) -> None:
        if self._white_hot is None:
            self._white_hot = detect_polarity_white_hot(first_gray)

    def step(self, frame_idx: int, frame_bgr: np.ndarray) -> StepResult:
        # First-frame polarity decision on raw grayscale.
        if self._white_hot is None:
            self._resolve_polarity(to_gray(frame_bgr))

        gray = preprocess(frame_bgr, self.cfg.preproc, self._white_hot)
        cands = detect(gray, self.cfg.detector)

        ts: TrackState_ | None = None
        if not self._initialised:
            if frame_idx >= self.cfg.init_frame_index:
                seed = self._seed_detection(cands)
                if seed is not None:
                    self.tracker.init(gray, seed)
                    self._initialised = True
                    ts = TrackState_(bbox=seed.bbox,
                                      state=TrackState.TRACKING,
                                      coast_frames=0, score=float(seed.score))
        else:
            ts = self.tracker.update(gray, cands)

        return StepResult(frame_idx=frame_idx, gray=gray,
                           candidates=cands, track=ts)

    def run(self, frames: Iterable[tuple[int, np.ndarray]]) -> Iterable[StepResult]:
        for idx, bgr in frames:
            yield self.step(idx, bgr)

    def _seed_detection(self, cands: list[Detection]) -> Detection | None:
        if self.cfg.init_bbox is not None:
            x, y, w, h = self.cfg.init_bbox
            return Detection(bbox=(int(x), int(y), int(w), int(h)),
                              score=1.0, area=int(w * h))
        return cands[0] if cands else None


# ---------------------------------------------------------------------------
# v2: motion-aware, fully unsupervised baseline
# ---------------------------------------------------------------------------


@dataclass
class MotionPipelineConfig:
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    detector: MotionDetectorConfig = field(default_factory=MotionDetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)

    warmup_frames: int = 5         # frames to fill the persistence map
    min_init_score: float = 40     # min mean_persistence * sqrt(area) to seed
    min_init_persistence_frames: int = 3   # blob must survive K consecutive frames
    init_streak_radius: float = 80.0       # px distance allowed between streak frames
    bad_diff_median: float = 30.0          # median(diff) above this → bad frame
    bad_diff_fraction: float = 0.30        # fraction(diff > 50) above this → bad


class MotionPipeline:
    """Motion-based detection + CSRT/Kalman tracker. Fully unsupervised.

    Per frame:
      1) preprocess (CLAHE + auto polarity); reset polarity on modality switch
      2) ORB+RANSAC homography prev→curr
      3) motion-compensated abs-diff → blend into warped persistence map
      4) threshold persistence → connected components → motion candidates
      5) once a candidate has been the top-ranked blob for K consecutive
         frames (after warmup), use it to init CSRT+Kalman
      6) thereafter, tracker runs as before, with motion candidates as
         the re-detection pool
    """

    def __init__(self, cfg: MotionPipelineConfig | None = None):
        self.cfg = cfg or MotionPipelineConfig()
        self._white_hot: bool | None = self.cfg.preproc.assume_white_hot
        self._prev_raw: np.ndarray | None = None   # raw gray, used for motion
        self._persistence = PersistenceMap(self.cfg.motion)
        self._modality = ModalityMonitor(self.cfg.modality)
        self.tracker = Tracker(self.cfg.tracker)
        self._initialised: bool = False
        self._candidate_streak: int = 0     # frames the top blob has been "good"
        self._last_top_center: tuple[float, float] | None = None

    # ----- lifecycle -----
    def _resolve_polarity(self, gray_raw: np.ndarray) -> None:
        if self._white_hot is None or self.cfg.preproc.assume_white_hot is None:
            self._white_hot = detect_polarity_white_hot(gray_raw)

    def _hard_reset_temporal(self) -> None:
        """Wipe persistence and tracker on a modality switch."""
        self._persistence.reset()
        self._prev_raw = None
        self._initialised = False
        self._candidate_streak = 0
        self._last_top_center = None
        self._white_hot = None
        self.tracker = Tracker(self.cfg.tracker)

    def _is_bad_diff(self, diff: np.ndarray) -> bool:
        med = float(np.median(diff))
        frac = float((diff > 50).mean())
        return med > self.cfg.bad_diff_median or frac > self.cfg.bad_diff_fraction

    # ----- per-frame -----
    def step(self, frame_idx: int, frame_bgr: np.ndarray) -> StepResult:
        raw_gray = to_gray(frame_bgr)
        switched = self._modality.step(raw_gray, frame_bgr)
        if switched:
            self._hard_reset_temporal()

        if self._white_hot is None:
            self._resolve_polarity(raw_gray)

        gray = preprocess(frame_bgr, self.cfg.preproc, self._white_hot)

        # First frame after reset → just seed history.
        if self._prev_raw is None:
            self._prev_raw = raw_gray
            return StepResult(frame_idx=frame_idx, gray=gray, candidates=[],
                              track=None, persistence=self._persistence.value(),
                              modality_switched=switched)

        # Ego-motion + compensated diff on RAW gray (CLAHE would itself drift).
        H = estimate_homography(self._prev_raw, raw_gray, self.cfg.motion)
        diff = compensated_diff(self._prev_raw, raw_gray, H, self.cfg.motion)
        self._prev_raw = raw_gray

        # Bad-diff guard (scene cut or warp failure): reset persistence
        # but keep tracker state and try again next frame.
        if self._is_bad_diff(diff):
            self._persistence.reset()
            pmap = self._persistence.value()
            return StepResult(frame_idx=frame_idx, gray=gray, candidates=[],
                              track=None, persistence=pmap,
                              modality_switched=switched)

        pmap = self._persistence.update(diff, H)

        # Detect on the persistence map.
        cands = detect_motion(pmap, self.cfg.detector)

        # Auto-init logic: require the top blob to persist near the same
        # location for several frames and exceed a score floor.
        ts: TrackState_ | None = None
        if not self._initialised:
            if (frame_idx >= self.cfg.warmup_frames and cands and
                    cands[0].score >= self.cfg.min_init_score):
                cx, cy = cands[0].center
                if self._last_top_center is not None:
                    dx = cx - self._last_top_center[0]
                    dy = cy - self._last_top_center[1]
                    if (dx * dx + dy * dy) ** 0.5 < self.cfg.init_streak_radius:
                        self._candidate_streak += 1
                    else:
                        self._candidate_streak = 1
                else:
                    self._candidate_streak = 1
                self._last_top_center = (cx, cy)

                if self._candidate_streak >= self.cfg.min_init_persistence_frames:
                    seed = cands[0]
                    self.tracker.init(gray, seed)
                    self._initialised = True
                    ts = TrackState_(bbox=seed.bbox,
                                      state=TrackState.TRACKING,
                                      coast_frames=0,
                                      score=float(seed.score))
            else:
                self._candidate_streak = 0
                self._last_top_center = None
        else:
            ts = self.tracker.update(gray, cands)

        return StepResult(frame_idx=frame_idx, gray=gray, candidates=cands,
                          track=ts, persistence=pmap,
                          modality_switched=switched)

    def run(self, frames: Iterable[tuple[int, np.ndarray]]) -> Iterable[StepResult]:
        for idx, bgr in frames:
            yield self.step(idx, bgr)
