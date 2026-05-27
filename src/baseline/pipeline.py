"""End-to-end baseline pipelines.

All pipelines share the same `StepResult` shape:

`Pipeline`              — v1 intensity-only baseline (top-hat + CSRT).
`MotionPipeline`        — v2 unsupervised motion-aware baseline.
`DLPipeline`            — phase 2: pretrained DL detector + same tracker.
`HybridPipeline`        — DL ∪ motion + CSRT/Kalman tracker w/ identity.
`FollowingPipeline`     — phase 2 pivot: detector-following design. DL
    picks "the person" each frame; Kalman only smooths + fills gaps.
    No CSRT, no appearance bank, no identity assumption beyond
    "closest detection to the last Kalman prediction."
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
from .tracker import (Tracker, TrackerConfig, TrackState, TrackState_,
                       _persistence_z)


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
                    self.tracker.init(gray, seed, frame_bgr=frame_bgr)
                    self._initialised = True
                    ts = TrackState_(bbox=seed.bbox,
                                      state=TrackState.TRACKING,
                                      coast_frames=0, score=float(seed.score))
        else:
            ts = self.tracker.update(gray, cands, frame_bgr=frame_bgr)

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

    warmup_frames: int = 5
    # A seed candidate must have persistence z-score > min_init_z and must
    # appear at roughly the same location for K consecutive frames. The
    # streak radius is expressed as a fraction of the frame diagonal so it
    # transfers across resolutions; no fixed pixel constant.
    min_init_z: float = 4.0
    min_init_persistence_frames: int = 3
    init_streak_radius_frac: float = 0.12  # of frame diagonal

    # "Bad diff" guard for scene cuts: anomaly cut against the rolling
    # median of recent per-frame diff statistics — not a fixed constant.
    bad_diff_window: int = 20
    bad_diff_k: float = 3.0
    bad_diff_fraction: float = 0.30


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
        self._candidate_streak: int = 0
        self._last_top_center: tuple[float, float] | None = None
        # Rolling stats for the bad-diff guard (data-driven, not video-tuned).
        self._diff_medians: list[float] = []
        self._diff_fracs: list[float] = []

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
        """Flag a frame as 'bad' (scene cut, severe warp failure) when its
        diff statistics are anomalous against the rolling history. No
        fixed pixel constant — purely robust statistics on this clip.
        """
        med = float(np.median(diff))
        frac = float((diff > 50).mean())
        bad = False
        if len(self._diff_medians) >= self.cfg.bad_diff_window:
            history = np.asarray(self._diff_medians[-self.cfg.bad_diff_window:])
            h_med = float(np.median(history))
            h_mad = float(np.median(np.abs(history - h_med))) * 1.4826
            thr = h_med + self.cfg.bad_diff_k * max(h_mad, 1.0)
            if med > thr or frac > self.cfg.bad_diff_fraction:
                bad = True
        self._diff_medians.append(med)
        self._diff_fracs.append(frac)
        if len(self._diff_medians) > 2 * self.cfg.bad_diff_window:
            self._diff_medians = self._diff_medians[-self.cfg.bad_diff_window:]
            self._diff_fracs = self._diff_fracs[-self.cfg.bad_diff_window:]
        return bad

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

        # Auto-init logic — data-driven, no video-specific constants:
        #   * top candidate's persistence z-score >= min_init_z
        #   * stays within `init_streak_radius_frac × frame_diagonal`
        #     for K consecutive frames
        ts: TrackState_ | None = None
        if not self._initialised:
            top_ok = False
            if frame_idx >= self.cfg.warmup_frames and cands:
                top = cands[0]
                if _persistence_z(pmap, top.bbox) >= self.cfg.min_init_z:
                    top_ok = True
            if top_ok:
                Hh, Ww = pmap.shape
                streak_radius = self.cfg.init_streak_radius_frac * (Ww * Ww + Hh * Hh) ** 0.5
                cx, cy = cands[0].center
                if self._last_top_center is not None:
                    dx = cx - self._last_top_center[0]
                    dy = cy - self._last_top_center[1]
                    if (dx * dx + dy * dy) ** 0.5 < streak_radius:
                        self._candidate_streak += 1
                    else:
                        self._candidate_streak = 1
                else:
                    self._candidate_streak = 1
                self._last_top_center = (cx, cy)

                if self._candidate_streak >= self.cfg.min_init_persistence_frames:
                    seed = cands[0]
                    self.tracker.init(gray, seed, frame_bgr=frame_bgr)
                    self._initialised = True
                    ts = TrackState_(bbox=seed.bbox,
                                      state=TrackState.TRACKING,
                                      coast_frames=0,
                                      score=float(seed.score))
            else:
                self._candidate_streak = 0
                self._last_top_center = None
        else:
            ts = self.tracker.update(gray, cands, persistence=pmap,
                                       ego_motion_H=H, frame_bgr=frame_bgr)

        return StepResult(frame_idx=frame_idx, gray=gray, candidates=cands,
                          track=ts, persistence=pmap,
                          modality_switched=switched)

    def run(self, frames: Iterable[tuple[int, np.ndarray]]) -> Iterable[StepResult]:
        for idx, bgr in frames:
            yield self.step(idx, bgr)


# ---------------------------------------------------------------------------
# phase 2: deep-learning detector + reused tracker
# ---------------------------------------------------------------------------


@dataclass
class DLPipelineConfig:
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    dl_detector: object | None = None      # DLDetectorConfig (typed lazily)
    init_min_conf: float = 0.25
    init_streak: int = 3
    init_streak_radius_frac: float = 0.12


class DLPipeline:
    """Per frame: run pretrained detector, feed candidates into the same
    Kalman + CSRT + Mahalanobis tracker the motion pipeline uses. No
    persistence map is computed → the tracker's persistence
    cross-validation is disabled automatically (appearance NCC alone
    gates measurements).
    """

    def __init__(self, cfg: DLPipelineConfig | None = None):
        # Lazy import so classical baselines do not require torch / ultralytics.
        from .dl_detector import DLDetector, DLDetectorConfig

        self.cfg = cfg or DLPipelineConfig()
        det_cfg = self.cfg.dl_detector or DLDetectorConfig()
        self.detector = DLDetector(det_cfg)
        self.tracker = Tracker(self.cfg.tracker)
        self._white_hot: bool | None = self.cfg.preproc.assume_white_hot
        self._initialised = False
        self._streak = 0
        self._last_center: tuple[float, float] | None = None

    def step(self, frame_idx: int, frame_bgr: np.ndarray) -> StepResult:
        raw_gray = to_gray(frame_bgr)
        if self._white_hot is None:
            self._white_hot = detect_polarity_white_hot(raw_gray)
        gray = preprocess(frame_bgr, self.cfg.preproc, self._white_hot)
        cands = self.detector(frame_bgr)

        ts: TrackState_ | None = None
        if not self._initialised:
            if cands and cands[0].score >= self.cfg.init_min_conf:
                Hh, Ww = gray.shape
                radius = self.cfg.init_streak_radius_frac * (Ww * Ww + Hh * Hh) ** 0.5
                cx, cy = cands[0].center
                if self._last_center is not None:
                    dx, dy = cx - self._last_center[0], cy - self._last_center[1]
                    self._streak = (self._streak + 1
                                     if (dx * dx + dy * dy) ** 0.5 < radius else 1)
                else:
                    self._streak = 1
                self._last_center = (cx, cy)
                if self._streak >= self.cfg.init_streak:
                    seed = cands[0]
                    self.tracker.init(gray, seed, frame_bgr=frame_bgr)
                    self._initialised = True
                    ts = TrackState_(bbox=seed.bbox,
                                      state=TrackState.TRACKING,
                                      coast_frames=0,
                                      score=float(seed.score))
            else:
                self._streak = 0
                self._last_center = None
        else:
            ts = self.tracker.update(gray, cands, persistence=None, frame_bgr=frame_bgr)

        return StepResult(frame_idx=frame_idx, gray=gray, candidates=cands,
                          track=ts, persistence=None, modality_switched=False)

    def run(self, frames: Iterable[tuple[int, np.ndarray]]) -> Iterable[StepResult]:
        for idx, bgr in frames:
            yield self.step(idx, bgr)


# ---------------------------------------------------------------------------
# phase 2 (hybrid): DL detector + motion persistence + everything else
# ---------------------------------------------------------------------------


@dataclass
class HybridPipelineConfig:
    """Hybrid = DL person prior + motion temporal evidence, full pipeline.

    Reuses every component built in phase 1 (modality monitor,
    persistence map, bad-diff guard, ego-motion homography) and adds the
    DL detector as a second candidate source. The tracker's persistence
    AND-gate cross-validation is *back on* because we again have a
    persistence map.
    """
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    detector_motion: MotionDetectorConfig = field(default_factory=MotionDetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    dl_detector: object | None = None      # DLDetectorConfig (typed lazily)

    # Init gate on the joint score (DL conf + λ·motion_z).
    min_init_joint: float = 0.20
    init_streak: int = 3
    init_streak_radius_frac: float = 0.12
    warmup_frames: int = 3

    motion_weight: float = 0.05

    # ByteTrack-style two-stage threshold. DL detections with raw
    # confidence above this are passed as "priority" candidates that
    # bypass the spatial gate.
    dl_priority_conf: float = 0.20

    bad_diff_window: int = 20
    bad_diff_k: float = 3.0
    bad_diff_fraction: float = 0.30


class HybridPipeline:
    """DL ∪ motion candidates, each cross-scored with motion-persistence
    z. Same Kalman + Mahalanobis + AND-gate tracker as the motion
    pipeline. Same modality reset. DL input is polarity-corrected
    (inverted on black-hot frames) so the white-hot-trained YOLO
    sees the polarity it expects.
    """

    def __init__(self, cfg: HybridPipelineConfig | None = None):
        from .dl_detector import DLDetector, DLDetectorConfig

        self.cfg = cfg or HybridPipelineConfig()
        self._modality = ModalityMonitor(self.cfg.modality)
        self._persistence = PersistenceMap(self.cfg.motion)
        self._prev_raw: np.ndarray | None = None
        self.dl = DLDetector(self.cfg.dl_detector or DLDetectorConfig())
        self.tracker = Tracker(self.cfg.tracker)
        self._white_hot: bool | None = self.cfg.preproc.assume_white_hot
        self._initialised = False
        self._streak = 0
        self._last_center: tuple[float, float] | None = None
        self._diff_medians: list[float] = []

    def _hard_reset_temporal(self) -> None:
        self._persistence.reset()
        self._prev_raw = None
        self._initialised = False
        self._streak = 0
        self._last_center = None
        self._white_hot = None
        self.tracker = Tracker(self.cfg.tracker)

    def _is_bad_diff(self, diff: np.ndarray) -> bool:
        med = float(np.median(diff))
        frac = float((diff > 50).mean())
        bad = False
        if len(self._diff_medians) >= self.cfg.bad_diff_window:
            history = np.asarray(self._diff_medians[-self.cfg.bad_diff_window:])
            h_med = float(np.median(history))
            h_mad = float(np.median(np.abs(history - h_med))) * 1.4826
            thr = h_med + self.cfg.bad_diff_k * max(h_mad, 1.0)
            if med > thr or frac > self.cfg.bad_diff_fraction:
                bad = True
        self._diff_medians.append(med)
        if len(self._diff_medians) > 2 * self.cfg.bad_diff_window:
            self._diff_medians = self._diff_medians[-self.cfg.bad_diff_window:]
        return bad

    def step(self, frame_idx: int, frame_bgr: np.ndarray) -> StepResult:
        raw_gray = to_gray(frame_bgr)
        switched = self._modality.step(raw_gray, frame_bgr)
        if switched:
            self._hard_reset_temporal()
        if self._white_hot is None:
            self._white_hot = detect_polarity_white_hot(raw_gray)
        gray = preprocess(frame_bgr, self.cfg.preproc, self._white_hot)

        # Motion path
        pmap: np.ndarray | None = None
        motion_cands: list[Detection] = []
        Hmat: np.ndarray | None = None
        if self._prev_raw is None:
            self._prev_raw = raw_gray
        else:
            Hmat = estimate_homography(self._prev_raw, raw_gray, self.cfg.motion)
            diff = compensated_diff(self._prev_raw, raw_gray, Hmat, self.cfg.motion)
            self._prev_raw = raw_gray
            if self._is_bad_diff(diff):
                self._persistence.reset()
            else:
                pmap = self._persistence.update(diff, Hmat)
                motion_cands = detect_motion(pmap, self.cfg.detector_motion)

        # DL path with polarity correction
        invert = not bool(self._white_hot)
        dl_cands = self.dl(frame_bgr, invert=invert)

        # Joint scoring
        fused: list[Detection] = []
        for d in dl_cands:
            z = _persistence_z(pmap, d.bbox) if pmap is not None else 0.0
            joint = float(d.score) + self.cfg.motion_weight * max(z, 0.0)
            fused.append(Detection(bbox=d.bbox, score=joint, area=d.area))
        for d in motion_cands:
            z = _persistence_z(pmap, d.bbox) if pmap is not None else 0.0
            base = float(np.tanh(max(z, 0.0) / 4.0))
            joint = base + self.cfg.motion_weight * max(z, 0.0)
            fused.append(Detection(bbox=d.bbox, score=joint, area=d.area))
        fused.sort(key=lambda c: c.score, reverse=True)

        ts: TrackState_ | None = None
        if not self._initialised:
            if (frame_idx >= self.cfg.warmup_frames and fused and
                    fused[0].score >= self.cfg.min_init_joint):
                Hh, Ww = gray.shape
                radius = self.cfg.init_streak_radius_frac * (Ww * Ww + Hh * Hh) ** 0.5
                cx, cy = fused[0].center
                if self._last_center is not None:
                    dx, dy = cx - self._last_center[0], cy - self._last_center[1]
                    self._streak = (self._streak + 1
                                     if (dx * dx + dy * dy) ** 0.5 < radius else 1)
                else:
                    self._streak = 1
                self._last_center = (cx, cy)
                if self._streak >= self.cfg.init_streak:
                    seed = fused[0]
                    self.tracker.init(gray, seed, frame_bgr=frame_bgr)
                    self._initialised = True
                    ts = TrackState_(bbox=seed.bbox,
                                      state=TrackState.TRACKING,
                                      coast_frames=0, score=float(seed.score))
            else:
                self._streak = 0
                self._last_center = None
        else:
            # ByteTrack-style: high-conf raw DL detections are
            # "priority" — accepted by appearance alone, no spatial
            # gate.
            priority = [d for d in dl_cands
                         if d.score >= self.cfg.dl_priority_conf]
            ts = self.tracker.update(gray, fused, persistence=pmap,
                                       ego_motion_H=Hmat,
                                       frame_bgr=frame_bgr,
                                       priority=priority)

        return StepResult(frame_idx=frame_idx, gray=gray, candidates=fused,
                          track=ts, persistence=pmap,
                          modality_switched=switched)

    def run(self, frames):
        for idx, bgr in frames:
            yield self.step(idx, bgr)


# ---------------------------------------------------------------------------
# phase 2 pivot: detector-following design
# ---------------------------------------------------------------------------


@dataclass
class FollowingPipelineConfig:
    """Detector-following pipeline.

    The DL detector chooses the target each frame; the Kalman filter
    only smooths the bbox and predicts during gaps where the detector
    misses. There is **no CSRT, no appearance template, no identity
    bank**. The motion-persistence stream remains as a *fallback
    candidate source* on frames where the DL detector fires nothing
    (e.g. colormapped IR segments).

    Identity is implicit: among the available candidates this frame,
    pick the one closest to the Kalman-predicted position. If no
    detection is close enough but a high-conf DL detection exists
    anywhere, accept it (the detector's person-class prior dominates).
    """
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    detector_motion: MotionDetectorConfig = field(default_factory=MotionDetectorConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    dl_detector: object | None = None      # DLDetectorConfig (typed lazily)

    # Selection
    dl_min_conf: float = 0.08              # minimum DL conf to consider
    dl_strong_conf: float = 0.15           # 'strong' DL: accepted anywhere
    motion_fallback_z: float = 4.0         # min motion z for fallback use
    proximity_radius_frac: float = 0.20    # of frame diagonal; preferred zone

    # Auto-init: a DL detection with conf >= dl_strong_conf seeds the
    # track. We deliberately do NOT allow motion-only init — motion
    # alone cannot answer "is this a person?".
    warmup_frames: int = 5

    # Coast
    max_coast_frames: int = 30

    # Bad-diff / scene-cut guard (same logic as MotionPipeline)
    bad_diff_window: int = 20
    bad_diff_k: float = 3.0
    bad_diff_fraction: float = 0.30


def _make_kalman_pos_only() -> "cv2.KalmanFilter":
    """Smaller Kalman: state = [cx, cy, w, h, vx, vy], meas = [cx, cy, w, h].
    Same as the identity tracker's filter — we reuse it but the rest of
    the FollowingPipeline ignores its appearance machinery."""
    import cv2 as _cv2
    return Tracker(TrackerConfig(use_reid=False)).kf  # cheap reuse


class FollowingPipeline:
    """DL-detector-following pipeline (post-step-D pivot)."""

    def __init__(self, cfg: FollowingPipelineConfig | None = None):
        from .dl_detector import DLDetector, DLDetectorConfig
        import cv2 as _cv2

        self.cfg = cfg or FollowingPipelineConfig()
        self._modality = ModalityMonitor(self.cfg.modality)
        self._persistence = PersistenceMap(self.cfg.motion)
        self._prev_raw: np.ndarray | None = None
        self.dl = DLDetector(self.cfg.dl_detector or DLDetectorConfig())

        # Kalman (we use the same _make_kalman from tracker.py implicitly
        # by instantiating a Tracker for its filter only; cheap.)
        from .tracker import _make_kalman, _bbox_to_meas, _state_to_bbox
        self._make_kalman = _make_kalman
        self._bbox_to_meas = _bbox_to_meas
        self._state_to_bbox = _state_to_bbox
        self.kf = self._make_kalman()

        self.state = TrackState.INIT
        self.bbox: tuple[int, int, int, int] | None = None
        self.coast_frames = 0
        self._white_hot: bool | None = self.cfg.preproc.assume_white_hot
        self._diff_medians: list[float] = []

    # ----- shared mechanics with MotionPipeline -----
    def _hard_reset_temporal(self) -> None:
        self._persistence.reset()
        self._prev_raw = None
        self._white_hot = None
        self.state = TrackState.INIT
        self.bbox = None
        self.coast_frames = 0
        self.kf = self._make_kalman()

    def _is_bad_diff(self, diff: np.ndarray) -> bool:
        med = float(np.median(diff))
        frac = float((diff > 50).mean())
        bad = False
        if len(self._diff_medians) >= self.cfg.bad_diff_window:
            hist = np.asarray(self._diff_medians[-self.cfg.bad_diff_window:])
            h_med = float(np.median(hist))
            h_mad = float(np.median(np.abs(hist - h_med))) * 1.4826
            thr = h_med + self.cfg.bad_diff_k * max(h_mad, 1.0)
            if med > thr or frac > self.cfg.bad_diff_fraction:
                bad = True
        self._diff_medians.append(med)
        if len(self._diff_medians) > 2 * self.cfg.bad_diff_window:
            self._diff_medians = self._diff_medians[-self.cfg.bad_diff_window:]
        return bad

    def _warp_kalman_by_homography(self, H) -> None:
        """Apply inter-frame homography to the Kalman state (same as
        Tracker._warp_state_by_homography)."""
        import cv2 as _cv2
        if H is None or self.bbox is None:
            return
        cx = float(self.kf.statePost[0, 0])
        cy = float(self.kf.statePost[1, 0])
        vx = float(self.kf.statePost[4, 0])
        vy = float(self.kf.statePost[5, 0])
        pts = np.array([[[cx, cy], [cx + vx, cy + vy]]], dtype=np.float32)
        warped = _cv2.perspectiveTransform(pts, H.astype(np.float32))
        ncx, ncy = float(warped[0, 0, 0]), float(warped[0, 0, 1])
        ntx, nty = float(warped[0, 1, 0]), float(warped[0, 1, 1])
        self.kf.statePost[0, 0] = ncx
        self.kf.statePost[1, 0] = ncy
        self.kf.statePost[4, 0] = ntx - ncx
        self.kf.statePost[5, 0] = nty - ncy

    def _pick_winner(self,
                      dl_cands: list[Detection],
                      motion_cands: list[Detection],
                      pmap: np.ndarray | None,
                      predicted_center: tuple[float, float] | None,
                      proximity_radius: float
                      ) -> Detection | None:
        """Selection rule for detector-following:

        1) If we have a prior Kalman estimate, prefer the DL candidate
           **closest** to it whose conf >= dl_min_conf. This is the
           identity-by-proximity rule.
        2) If none is close enough, fall back to the **highest-conf**
           DL candidate anywhere with conf >= dl_strong_conf.
        3) If still nothing, fall back to the motion candidate with
           highest persistence z above motion_fallback_z (only if no
           prior, or proximal to it).
        4) Otherwise return None (→ COASTING).
        """
        dl_good = [d for d in dl_cands if d.score >= self.cfg.dl_min_conf]

        # Rule 1: closest DL to prediction
        if predicted_center is not None and dl_good:
            cands_with_dist = []
            for d in dl_good:
                cx, cy = d.center
                dx, dy = cx - predicted_center[0], cy - predicted_center[1]
                cands_with_dist.append(((dx * dx + dy * dy) ** 0.5, d))
            cands_with_dist.sort(key=lambda t: t[0])
            d_closest, det_closest = cands_with_dist[0]
            if d_closest <= proximity_radius:
                return det_closest

        # Rule 2: strongest DL anywhere
        strong = [d for d in dl_cands if d.score >= self.cfg.dl_strong_conf]
        if strong:
            strong.sort(key=lambda d: -d.score)
            return strong[0]

        # Rule 3: motion fallback — only when we already have a track
        #         (motion alone cannot tell "is this a person?", so we
        #          refuse to use it for the initial identity).
        if predicted_center is not None and motion_cands and pmap is not None:
            from .tracker import _persistence_z
            scored = []
            for d in motion_cands:
                z = _persistence_z(pmap, d.bbox)
                if z >= self.cfg.motion_fallback_z:
                    scored.append((z, d))
            if scored:
                scored.sort(key=lambda t: (
                    (t[1].center[0] - predicted_center[0]) ** 2 +
                    (t[1].center[1] - predicted_center[1]) ** 2))
                _, best_motion = scored[0]
                dx = best_motion.center[0] - predicted_center[0]
                dy = best_motion.center[1] - predicted_center[1]
                if (dx * dx + dy * dy) ** 0.5 <= proximity_radius:
                    return best_motion

        return None

    def step(self, frame_idx: int, frame_bgr: np.ndarray) -> StepResult:
        raw_gray = to_gray(frame_bgr)
        switched = self._modality.step(raw_gray, frame_bgr)
        if switched:
            self._hard_reset_temporal()
        if self._white_hot is None:
            self._white_hot = detect_polarity_white_hot(raw_gray)
        gray = preprocess(frame_bgr, self.cfg.preproc, self._white_hot)

        # Ego-motion + persistence (for fallback candidates only)
        pmap: np.ndarray | None = None
        motion_cands: list[Detection] = []
        Hmat = None
        if self._prev_raw is None:
            self._prev_raw = raw_gray
        else:
            Hmat = estimate_homography(self._prev_raw, raw_gray, self.cfg.motion)
            diff = compensated_diff(self._prev_raw, raw_gray, Hmat, self.cfg.motion)
            self._prev_raw = raw_gray
            if self._is_bad_diff(diff):
                self._persistence.reset()
            else:
                pmap = self._persistence.update(diff, Hmat)
                motion_cands = detect_motion(pmap, self.cfg.detector_motion)

        # DL detections, polarity-corrected
        invert = not bool(self._white_hot)
        dl_cands = self.dl(frame_bgr, invert=invert)

        # Camera-motion warp the Kalman state (if we have a track)
        if Hmat is not None and self.state in (TrackState.TRACKING,
                                                TrackState.COASTING) \
                and self.bbox is not None:
            self._warp_kalman_by_homography(Hmat)

        # Predict and snapshot prior position
        Hh, Ww = gray.shape
        diag = (Hh * Hh + Ww * Ww) ** 0.5
        radius = self.cfg.proximity_radius_frac * diag
        if self.bbox is not None:
            pred = self.kf.predict()
            pred_bbox = self._state_to_bbox(pred[:4])
            pcx = float(self.kf.statePre[0, 0])
            pcy = float(self.kf.statePre[1, 0])
            predicted_center = (pcx, pcy)
        else:
            pred_bbox = None
            predicted_center = None

        # Pick the winner
        winner = self._pick_winner(dl_cands, motion_cands, pmap,
                                    predicted_center, radius)

        # Pre-warmup window: don't init yet
        if winner is not None and frame_idx < self.cfg.warmup_frames \
                and self.bbox is None:
            winner = None

        # Apply
        ts: TrackState_ | None = None
        if winner is not None:
            meas = self._bbox_to_meas(winner.bbox)
            if self.bbox is None:
                # Init: snap Kalman to the detection with zero velocity.
                self.kf.statePost = np.vstack(
                    [meas, np.zeros((2, 1), dtype=np.float32)])
                self.kf.statePre = self.kf.statePost.copy()
            else:
                self.kf.correct(meas)
            self.bbox = self._state_to_bbox(self.kf.statePost[:4])
            self.state = TrackState.TRACKING
            self.coast_frames = 0
            ts = TrackState_(bbox=self.bbox, state=self.state,
                              coast_frames=0, score=float(winner.score))
        elif self.bbox is not None:
            # Coast on Kalman prediction
            self.bbox = pred_bbox if pred_bbox is not None else self.bbox
            self.coast_frames += 1
            self.state = (TrackState.LOST
                          if self.coast_frames > self.cfg.max_coast_frames
                          else TrackState.COASTING)
            ts = TrackState_(bbox=self.bbox, state=self.state,
                              coast_frames=self.coast_frames, score=0.0)

        # Combined candidate list for visualisation
        all_cands = list(dl_cands) + list(motion_cands)
        return StepResult(frame_idx=frame_idx, gray=gray, candidates=all_cands,
                          track=ts, persistence=pmap,
                          modality_switched=switched)

    def run(self, frames):
        for idx, bgr in frames:
            yield self.step(idx, bgr)


# ---------------------------------------------------------------------------
# phase 3: multi-object identity tracker with ReID (BoT-SORT-style)
# ---------------------------------------------------------------------------


@dataclass
class IDPipelineConfig:
    """Multi-track pipeline.

    YOLO detects everything each frame; IdentityTracker assigns one
    persistent integer ID per object via a three-stage cascade
    (IoU → appearance → lost-pool re-ID). Camera-motion compensation
    and modality switching are inherited from the rest of the codebase.
    """
    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    dl_detector: object | None = None
    id_tracker: object | None = None      # IdentityTrackerConfig


@dataclass
class IDStepResult:
    """Per-frame output for the multi-track pipeline.

    Mirrors `StepResult` but carries a *list* of tracks instead of one.
    """
    frame_idx: int
    gray: np.ndarray
    candidates: list[Detection]
    tracks: list                          # list[IdentityTrack]
    persistence: np.ndarray | None = None
    modality_switched: bool = False


class IDPipeline:
    """YOLO + IdentityTracker. The detector decides "what's there";
    the IdentityTracker assigns persistent IDs across frames including
    re-identification after the target temporarily disappears.
    """

    def __init__(self, cfg: IDPipelineConfig | None = None):
        from .dl_detector import DLDetector, DLDetectorConfig
        from .id_tracker import IdentityTracker, IdentityTrackerConfig

        self.cfg = cfg or IDPipelineConfig()
        self._modality = ModalityMonitor(self.cfg.modality)
        self._prev_raw: np.ndarray | None = None
        self.dl = DLDetector(self.cfg.dl_detector or DLDetectorConfig())
        self.id_tracker = IdentityTracker(
            self.cfg.id_tracker or IdentityTrackerConfig())
        self._white_hot: bool | None = self.cfg.preproc.assume_white_hot

    def step(self, frame_idx: int, frame_bgr: np.ndarray) -> IDStepResult:
        raw_gray = to_gray(frame_bgr)
        switched = self._modality.step(raw_gray, frame_bgr)
        if switched:
            self.id_tracker.reset()
            self._prev_raw = None
            self._white_hot = None
        if self._white_hot is None:
            self._white_hot = detect_polarity_white_hot(raw_gray)
        gray = preprocess(frame_bgr, self.cfg.preproc, self._white_hot)

        # Ego-motion homography (cheap; reused for the Kalman warp).
        Hmat = None
        if self._prev_raw is not None:
            Hmat = estimate_homography(self._prev_raw, raw_gray, self.cfg.motion)
        self._prev_raw = raw_gray

        # DL detections.
        # Empirical: bit-inverting the input to YOLO on "black-hot"
        # frames *hurts* recall on this checkpoint (the model is
        # apparently robust to either polarity, and our inversion
        # introduces texture artefacts that look like edge-of-image
        # bodies). We pass the raw frame.
        dl_cands = self.dl(frame_bgr, invert=False)

        # Multi-track update.
        active_tracks = self.id_tracker.step(
            dl_cands, gray, frame_bgr, ego_motion_H=Hmat)

        return IDStepResult(
            frame_idx=frame_idx,
            gray=gray,
            candidates=dl_cands,
            tracks=list(active_tracks),
            persistence=None,
            modality_switched=switched,
        )

    def run(self, frames):
        for idx, bgr in frames:
            yield self.step(idx, bgr)
