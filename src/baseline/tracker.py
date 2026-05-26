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
    """Tracker hyperparameters.

    None of these are tuned to a specific clip; they encode standard
    Bayesian-tracking principles:

      - `gate_sigma` is a Mahalanobis chi-square radius (sigma multiple).
      - `min_appearance_ncc` is the standard NCC cut for visual match.
      - `min_persistence_z` is a z-score (vs frame-wide median+MAD) for
        independent motion evidence.

    A measurement is only accepted if **both** appearance and motion
    evidence agree (cross-validated AND-gate). The candidate-association
    gate widens automatically as the filter's own position covariance
    grows during coast — no hand-set radius.
    """
    max_coast_frames: int = 30
    gate_sigma: float = 3.0           # Mahalanobis χ radius
    gate_min_radius_px: float = 12.0  # floor (we never trust a sub-pixel gate)
    min_appearance_ncc: float = 0.30
    min_persistence_z: float = 2.0
    appearance_alpha: float = 0.2


@dataclass
class TrackState_:
    bbox: tuple[int, int, int, int]
    state: TrackState
    coast_frames: int = 0
    score: float = 1.0
    appearance: np.ndarray | None = field(default=None, repr=False)


def _make_kalman() -> cv2.KalmanFilter:
    # State: [cx, cy, w, h, vx, vy], measurement: [cx, cy, w, h].
    # Q (process noise) is calibrated so position uncertainty grows by a
    # few pixels per coast frame and velocity uncertainty grows faster:
    # after ~10 coast frames the predicted velocity carries almost no
    # information, which is what we want when measurements stop.
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
    kf.processNoiseCov = np.diag([4.0, 4.0, 1.0, 1.0, 9.0, 9.0]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([1.0, 1.0, 4.0, 4.0]).astype(np.float32)
    kf.errorCovPost = np.diag([10.0, 10.0, 10.0, 10.0,
                                100.0, 100.0]).astype(np.float32)
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


def _persistence_z(persistence: np.ndarray | None,
                    bbox: tuple[int, int, int, int]) -> float:
    """Z-score of mean(persistence inside bbox) vs the global (median, MAD)
    of the persistence map. Robust to a few hot blobs because we use MAD
    rather than std. Returns 0 if no map is available.
    """
    if persistence is None or persistence.size == 0:
        return 0.0
    H, W = persistence.shape
    x, y, w, h = bbox
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inner_mean = float(persistence[y0:y1, x0:x1].mean())
    flat = persistence.ravel()
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med))) * 1.4826  # → std-equivalent
    if mad < 1e-3:
        return 0.0
    return (inner_mean - med) / mad


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
        self._reseed_kalman(det.bbox)
        self._csrt = cv2.TrackerCSRT_create()
        self._csrt.init(gray, det.bbox)
        self.appearance = _appearance_patch(gray, det.bbox)
        self.bbox = det.bbox
        self.state = TrackState.TRACKING
        self.coast_frames = 0
        self.last_score = float(det.score)

    def update(self, gray: np.ndarray, candidates: list[Detection],
               persistence: np.ndarray | None = None) -> TrackState_:
        if self.state == TrackState.INIT:
            raise RuntimeError("Tracker.init() must be called first")

        # 1) Predict (also advances errorCovPre, which we use for the gate).
        pred = self.kf.predict()
        pred_bbox = _state_to_bbox(pred[:4])

        # 2) Try CSRT in the current frame.
        csrt_ok, csrt_bbox = False, None
        if self.state in (TrackState.TRACKING, TrackState.COASTING) and self._csrt is not None:
            csrt_ok, raw = self._csrt.update(gray)
            if csrt_ok:
                x, y, w, h = raw
                csrt_bbox = (int(x), int(y), int(w), int(h))

        # 3) AND-gate the CSRT measurement: it must satisfy BOTH
        #    appearance (NCC against stored model) AND independent motion
        #    evidence (persistence z-score above background).
        meas_bbox: tuple[int, int, int, int] | None = None
        score = 0.0
        if csrt_ok and csrt_bbox is not None:
            app = (_ncc(_appearance_patch(gray, csrt_bbox), self.appearance)
                   if self.appearance is not None else 0.0)
            p_z = _persistence_z(persistence, csrt_bbox)
            if app >= self.cfg.min_appearance_ncc and p_z >= self.cfg.min_persistence_z:
                meas_bbox = csrt_bbox
                score = app
                self.state = TrackState.TRACKING
            else:
                score = app  # remember for diagnostics

        # 4) If CSRT was rejected (or never ran), look in the Mahalanobis
        #    gate of the *current* covariance. Gate widens automatically
        #    during coast — no explicit "drop gate" branch.
        if meas_bbox is None:
            meas_bbox = self._mahalanobis_search(gray, candidates, persistence)
            if meas_bbox is not None:
                # Treat as a fresh acquisition: reset CSRT and velocity.
                self._reseed_kalman(meas_bbox)
                self._csrt = cv2.TrackerCSRT_create()
                self._csrt.init(gray, meas_bbox)
                self.state = TrackState.TRACKING
                self.coast_frames = 0
                score = max(score, 0.5)

        # 5) Update model on success; otherwise coast / declare lost.
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
    def _reseed_kalman(self, bbox: tuple[int, int, int, int]) -> None:
        """Snap the Kalman state to a measurement with **zero velocity** and
        reset the posterior covariance. Used on every (re-)acquisition so
        we never carry stale velocity through a re-init."""
        meas = _bbox_to_meas(bbox)
        self.kf.statePost = np.vstack([meas, np.zeros((2, 1), dtype=np.float32)])
        self.kf.statePre = self.kf.statePost.copy()
        self.kf.errorCovPost = np.diag([10.0, 10.0, 10.0, 10.0,
                                         100.0, 100.0]).astype(np.float32)
        self.kf.errorCovPre = self.kf.errorCovPost.copy()

    def _mahalanobis_search(self, gray: np.ndarray,
                             candidates: list[Detection],
                             persistence: np.ndarray | None
                             ) -> tuple[int, int, int, int] | None:
        """Pick the candidate that is statistically consistent with the
        Kalman prediction and also passes the appearance + persistence
        evidence test. Gate radius is derived from the filter's own
        position covariance, so it widens during coast automatically.
        """
        if not candidates:
            return None
        cov = self.kf.errorCovPre[:2, :2].astype(np.float64)
        try:
            cov_inv = np.linalg.inv(cov + 1e-3 * np.eye(2))
        except np.linalg.LinAlgError:
            cov_inv = np.eye(2)
        pcx = float(self.kf.statePre[0, 0])
        pcy = float(self.kf.statePre[1, 0])
        chi2_gate = float(self.cfg.gate_sigma) ** 2

        # Always allow a small absolute-pixel floor on the gate so we can
        # accept measurements when the filter momentarily thinks it is
        # very certain (just after a correction).
        sx = float(np.sqrt(max(cov[0, 0], 0.0)))
        sy = float(np.sqrt(max(cov[1, 1], 0.0)))
        min_r = float(self.cfg.gate_min_radius_px)

        best: tuple[float, tuple[int, int, int, int]] | None = None
        for d in candidates:
            cx, cy = d.center
            dx, dy = cx - pcx, cy - pcy
            mahal2 = float(dx * dx * cov_inv[0, 0]
                            + 2 * dx * dy * cov_inv[0, 1]
                            + dy * dy * cov_inv[1, 1])
            dist = (dx * dx + dy * dy) ** 0.5
            radius_floor = max(min_r, self.cfg.gate_sigma * max(sx, sy))
            if mahal2 > chi2_gate and dist > radius_floor:
                continue
            # Appearance + persistence cross-validation. We allow either
            # signal to be borderline as long as the other is strong.
            app = (_ncc(_appearance_patch(gray, d.bbox), self.appearance)
                   if self.appearance is not None else 0.0)
            p_z = _persistence_z(persistence, d.bbox)
            if app < self.cfg.min_appearance_ncc and p_z < self.cfg.min_persistence_z:
                continue
            # Joint score — appearance plus mild bonus for stronger motion
            # evidence and a small penalty for Mahalanobis distance.
            s = app + 0.05 * p_z - 0.02 * mahal2
            if best is None or s > best[0]:
                best = (s, d.bbox)
        return best[1] if best else None
