"""Single-target tracker: CSRT short-term + Kalman coast + re-detection gate."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np

from .detector import Detection


class TrackState(Enum):
    INIT = auto()
    TRACKING = auto()
    COASTING = auto()    # CSRT lost; Kalman predicting only
    LOST = auto()


@dataclass
class TrackerConfig:
    max_coast_frames: int = 30        # how long Kalman may coast before LOST
    redetect_gate_scale: float = 3.0  # search radius = scale * max(w,h)
    min_redetect_iou: float = 0.0     # accept any candidate inside the gate
    appearance_alpha: float = 0.2     # EMA on stored grayscale patch


@dataclass
class TrackState_:
    bbox: tuple[int, int, int, int]
    state: TrackState
    coast_frames: int = 0
    score: float = 1.0
    appearance: np.ndarray | None = field(default=None, repr=False)


def _make_kalman() -> cv2.KalmanFilter:
    # State: [cx, cy, w, h, vx, vy], measurement: [cx, cy, w, h].
    kf = cv2.KalmanFilter(6, 4)
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
    kf.processNoiseCov = np.diag([1, 1, 1, 1, 4, 4]).astype(np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf


def _bbox_to_meas(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([[x + w / 2.0], [y + h / 2.0], [w], [h]], dtype=np.float32)


def _state_to_bbox(state: np.ndarray) -> tuple[int, int, int, int]:
    cx, cy, w, h = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
    w = max(2.0, float(w)); h = max(2.0, float(h))
    return int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h))


def _appearance_patch(gray: np.ndarray, bbox: tuple[int, int, int, int],
                      out_size: int = 32) -> np.ndarray:
    x, y, w, h = bbox
    H, W = gray.shape
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((out_size, out_size), dtype=np.float32)
    crop = gray[y0:y1, x0:x1]
    crop = cv2.resize(crop, (out_size, out_size)).astype(np.float32)
    crop -= crop.mean()
    n = np.linalg.norm(crop) + 1e-6
    return crop / n


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return -1.0
    return float((a * b).sum())


class Tracker:
    """Single-target tracker. Call init() once with a detection, then
    update() per frame with the next preprocessed frame and the current
    list of candidate detections (used for re-acquisition when coasting).
    """

    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.kf = _make_kalman()
        self._csrt: cv2.Tracker | None = None
        self.state = TrackState.INIT
        self.bbox: tuple[int, int, int, int] | None = None
        self.coast_frames = 0
        self.appearance: np.ndarray | None = None
        self.last_score: float = 0.0

    # ----- public API -----
    def init(self, gray: np.ndarray, det: Detection) -> None:
        self.bbox = det.bbox
        meas = _bbox_to_meas(det.bbox)
        self.kf.statePost = np.vstack([meas, np.zeros((2, 1), dtype=np.float32)])
        self.kf.statePre = self.kf.statePost.copy()
        self._csrt = cv2.TrackerCSRT_create()
        self._csrt.init(gray, det.bbox)
        self.appearance = _appearance_patch(gray, det.bbox)
        self.state = TrackState.TRACKING
        self.coast_frames = 0
        self.last_score = float(det.score)

    def update(self, gray: np.ndarray,
               candidates: list[Detection]) -> TrackState_:
        if self.state == TrackState.INIT:
            raise RuntimeError("Tracker.init() must be called first")

        # 1. Kalman predict.
        pred = self.kf.predict()
        pred_bbox = _state_to_bbox(pred[:4])

        # 2. Try CSRT.
        csrt_ok, csrt_bbox = False, None
        if self.state in (TrackState.TRACKING, TrackState.COASTING) and self._csrt is not None:
            csrt_ok, raw = self._csrt.update(gray)
            if csrt_ok:
                x, y, w, h = raw
                csrt_bbox = (int(x), int(y), int(w), int(h))

        # 3. Score CSRT output (NCC against stored appearance).
        meas_bbox: tuple[int, int, int, int] | None = None
        score = 0.0
        if csrt_ok and csrt_bbox is not None:
            patch = _appearance_patch(gray, csrt_bbox)
            score = _ncc(patch, self.appearance) if self.appearance is not None else 1.0
            if score > 0.3:                 # healthy match
                meas_bbox = csrt_bbox
                self.state = TrackState.TRACKING
            # else fall through to re-detect

        # 4. Re-detect inside Kalman-predicted gate if CSRT failed/weak.
        if meas_bbox is None:
            meas_bbox = self._search_in_gate(gray, pred_bbox, candidates)
            if meas_bbox is not None:
                # Re-init CSRT on the recovered location.
                self._csrt = cv2.TrackerCSRT_create()
                self._csrt.init(gray, meas_bbox)
                self.state = TrackState.TRACKING
                self.coast_frames = 0
                score = max(score, 0.5)

        # 5. Update Kalman + appearance, or coast.
        if meas_bbox is not None:
            self.kf.correct(_bbox_to_meas(meas_bbox))
            self.bbox = meas_bbox
            self.coast_frames = 0
            patch = _appearance_patch(gray, meas_bbox)
            if self.appearance is None:
                self.appearance = patch
            else:
                a = self.cfg.appearance_alpha
                blended = (1 - a) * self.appearance + a * patch
                n = np.linalg.norm(blended) + 1e-6
                self.appearance = blended / n
        else:
            self.coast_frames += 1
            self.bbox = pred_bbox
            self.state = (TrackState.LOST
                          if self.coast_frames > self.cfg.max_coast_frames
                          else TrackState.COASTING)

        self.last_score = float(score)
        return TrackState_(bbox=self.bbox, state=self.state,
                            coast_frames=self.coast_frames, score=self.last_score,
                            appearance=self.appearance)

    # ----- helpers -----
    def _search_in_gate(self, gray: np.ndarray,
                        pred_bbox: tuple[int, int, int, int],
                        candidates: list[Detection]
                        ) -> tuple[int, int, int, int] | None:
        if not candidates:
            return None
        px, py, pw, ph = pred_bbox
        pcx, pcy = px + pw / 2.0, py + ph / 2.0
        radius = self.cfg.redetect_gate_scale * max(pw, ph)

        best: tuple[float, Detection] | None = None
        for d in candidates:
            cx, cy = d.center
            dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            if dist > radius:
                continue
            # Score = appearance match minus a mild distance penalty.
            patch = _appearance_patch(gray, d.bbox)
            app = _ncc(patch, self.appearance) if self.appearance is not None else 0.0
            penalty = dist / max(radius, 1.0)
            s = app - 0.5 * penalty
            if best is None or s > best[0]:
                best = (s, d)
        if best is None:
            return None
        return best[1].bbox
