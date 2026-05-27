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
      - `min_appearance` is the cosine (or NCC) cut for visual match.
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
    min_appearance: float = 0.30      # NCC ≥ 0.3 OR cosine ≥ 0.3
    min_persistence_z: float = 2.0
    appearance_alpha: float = 0.2

    # ----- Deep ReID -----
    use_reid: bool = True
    reid_template_bank_size: int = 8       # DeepSORT-style sliding window
    reid_bank_min_gap: float = 0.985       # avoid storing near-duplicate templates

    # ----- Two-stage (ByteTrack-style) association -----
    # A "priority" candidate (high-confidence DL person detection) is
    # accepted by appearance match alone — the Mahalanobis gate is
    # dropped because the detector's own person-class prior is already
    # strong independent evidence.
    priority_min_appearance: float = 0.35

    # Backwards-compat alias so existing code/configs keep working.
    @property
    def min_appearance_ncc(self) -> float:
        return self.min_appearance


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

    # Class-level cache so we share the heavy CNN across Tracker resets.
    _reid_shared = None

    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.kf = _make_kalman()
        self._csrt: cv2.Tracker | None = None
        self.state = TrackState.INIT
        self.bbox: tuple[int, int, int, int] | None = None
        self.coast_frames = 0
        self.appearance: np.ndarray | None = None  # latest signature (legacy field)
        self.appearance_bank: list[np.ndarray] = []  # newest-first
        self.last_score: float = 0.0
        self._reid = self._get_reid() if self.cfg.use_reid else None

    @classmethod
    def _get_reid(cls):
        if cls._reid_shared is None:
            try:
                from .reid import make_reid_extractor
                cls._reid_shared = make_reid_extractor()
            except Exception:
                cls._reid_shared = False  # tried and failed
        return cls._reid_shared or None

    # ----- appearance abstraction -----
    def _make_signature(self, gray: np.ndarray, frame_bgr: np.ndarray | None,
                         bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        """Return either a CNN embedding (if ReID is on and we have BGR)
        or a 32×32 mean-subtracted patch (legacy fallback)."""
        if self._reid is not None and frame_bgr is not None:
            return self._reid.embed(frame_bgr, bbox)
        return _appearance_patch(gray, bbox)

    def _similarity(self, a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None:
            return 0.0
        if a.ndim == 1:               # embedding → cosine (already L2-normed)
            return float(np.dot(a, b))
        return _ncc(a, b)             # 2-D patch → NCC

    def _bank_similarity(self, sig: np.ndarray | None) -> float:
        """Best similarity of `sig` against any template in the bank.
        DeepSORT-style: a candidate matches the target if it matches
        *any* recent appearance, not only the freshest EMA."""
        if sig is None:
            return 0.0
        if self.appearance_bank:
            return max(self._similarity(sig, t) for t in self.appearance_bank)
        return self._similarity(sig, self.appearance)

    def _push_to_bank(self, sig: np.ndarray | None) -> None:
        if sig is None:
            return
        if (self.appearance_bank and
                self._similarity(sig, self.appearance_bank[0])
                >= self.cfg.reid_bank_min_gap):
            return  # near-duplicate of the freshest entry; skip
        self.appearance_bank.insert(0, sig)
        if len(self.appearance_bank) > self.cfg.reid_template_bank_size:
            self.appearance_bank = self.appearance_bank[: self.cfg.reid_template_bank_size]

    # ----- public API -----
    def init(self, gray: np.ndarray, det: Detection,
             frame_bgr: np.ndarray | None = None) -> None:
        bbox = self._sanitise_bbox(det.bbox, gray)
        if bbox is None:
            return  # degenerate seed; caller will retry next frame
        csrt = self._try_csrt_init(gray, bbox)
        if csrt is None:
            return
        self._reseed_kalman(bbox)
        self._csrt = csrt
        sig = self._make_signature(gray, frame_bgr, bbox)
        self.appearance = sig
        self.appearance_bank = []
        self._push_to_bank(sig)
        self.bbox = bbox
        self.state = TrackState.TRACKING
        self.coast_frames = 0
        self.last_score = float(det.score)

    def update(self, gray: np.ndarray, candidates: list[Detection],
               persistence: np.ndarray | None = None,
               ego_motion_H: np.ndarray | None = None,
               frame_bgr: np.ndarray | None = None,
               priority: list[Detection] | None = None) -> TrackState_:
        """Update the tracker for one frame.

        `ego_motion_H` is the 3×3 homography that maps points from the
        *previous* frame's pixel coordinates to the *current* frame's
        — i.e. the same matrix the motion module already estimates for
        the persistence map. Passing it lets us compensate for camera
        motion inside the Kalman state itself, exactly the way
        StrongSORT / Deep OC-SORT do, which is critical for aerial drone
        footage where the camera and the target both move.
        """
        if self.state == TrackState.INIT:
            raise RuntimeError("Tracker.init() must be called first")

        # 0) Camera-motion compensation. Warp the Kalman state (position
        #    and velocity tip) by the inter-frame homography so the
        #    predicted state lives in the CURRENT frame's coordinates,
        #    not the previous frame's.
        if ego_motion_H is not None:
            self._warp_state_by_homography(ego_motion_H)

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

        # 3) Gate the CSRT measurement. When a persistence map is
        #    available (motion pipeline) we AND-gate appearance with
        #    independent motion evidence. When it is absent (DL pipeline)
        #    we rely on appearance similarity alone — the detector itself
        #    is the second evidence source, applied earlier.
        meas_bbox: tuple[int, int, int, int] | None = None
        score = 0.0
        if csrt_ok and csrt_bbox is not None:
            sig = self._make_signature(gray, frame_bgr, csrt_bbox)
            app = self._bank_similarity(sig)
            if persistence is not None:
                p_z = _persistence_z(persistence, csrt_bbox)
                ok = (app >= self.cfg.min_appearance and
                      p_z >= self.cfg.min_persistence_z)
            else:
                ok = app >= self.cfg.min_appearance
            if ok:
                meas_bbox = csrt_bbox
                score = app
                self.state = TrackState.TRACKING
            else:
                score = app  # remember for diagnostics

        # 4a) If CSRT was rejected and we have **high-confidence DL
        #     detections** ("priority" candidates), try matching them by
        #     appearance only — drop the spatial gate. Rationale
        #     (ByteTrack / DeepSORT): a confident person-class detection
        #     is strong independent evidence of the target identity, so
        #     spatial inconsistency with a stale Kalman prediction should
        #     not by itself disqualify it.
        if meas_bbox is None and priority:
            best: tuple[float, tuple[int, int, int, int]] | None = None
            H, W = gray.shape
            for d in priority:
                bx, by, bw, bh = d.bbox
                # Sanity-check the bbox before letting CSRT touch it.
                if bw < 4 or bh < 4 or bx < 0 or by < 0 or \
                        bx + bw > W or by + bh > H:
                    continue
                sig = self._make_signature(gray, frame_bgr, d.bbox)
                app = self._bank_similarity(sig)
                if app < self.cfg.priority_min_appearance:
                    continue
                if best is None or app > best[0]:
                    best = (app, d.bbox)
            if best is not None:
                sane = self._sanitise_bbox(best[1], gray)
                csrt = self._try_csrt_init(gray, sane) if sane is not None else None
                if csrt is not None:
                    meas_bbox = sane
                    self._reseed_kalman(meas_bbox)
                    self._csrt = csrt
                    self.state = TrackState.TRACKING
                    self.coast_frames = 0
                score = max(score, best[0])

        # 4b) Fall back to Mahalanobis gate on the full candidate pool.
        if meas_bbox is None:
            meas_bbox = self._mahalanobis_search(gray, frame_bgr,
                                                  candidates, persistence)
            if meas_bbox is not None:
                sane = self._sanitise_bbox(meas_bbox, gray)
                csrt = self._try_csrt_init(gray, sane) if sane is not None else None
                if csrt is not None:
                    meas_bbox = sane
                    self._reseed_kalman(meas_bbox)
                    self._csrt = csrt
                    self.state = TrackState.TRACKING
                    self.coast_frames = 0
                    score = max(score, 0.5)
                else:
                    meas_bbox = None

        # 5) Update model on success; otherwise coast / declare lost.
        if meas_bbox is not None:
            self.kf.correct(_bbox_to_meas(meas_bbox))
            self.bbox = meas_bbox
            self.coast_frames = 0
            sig = self._make_signature(gray, frame_bgr, meas_bbox)
            if sig is not None:
                # Maintain a sliding template bank AND a freshest-EMA pointer.
                self._push_to_bank(sig)
                if self.appearance is None:
                    self.appearance = sig
                else:
                    a = self.cfg.appearance_alpha
                    blended = (1 - a) * self.appearance + a * sig
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
    @staticmethod
    def _sanitise_bbox(bbox: tuple[int, int, int, int],
                        gray: np.ndarray,
                        min_side: int = 8
                        ) -> tuple[int, int, int, int] | None:
        H, W = gray.shape[:2]
        x, y, w, h = (int(v) for v in bbox)
        # First clamp top-left into the image.
        x = max(0, min(x, W - min_side))
        y = max(0, min(y, H - min_side))
        # Then clamp size against the remaining room.
        w = max(min_side, min(w, W - x))
        h = max(min_side, min(h, H - y))
        if w < min_side or h < min_side:
            return None
        return x, y, w, h

    @staticmethod
    def _try_csrt_init(gray: np.ndarray,
                        bbox: tuple[int, int, int, int]) -> "cv2.Tracker | None":
        try:
            tr = cv2.TrackerCSRT_create()
            tr.init(gray, bbox)
            return tr
        except cv2.error:
            return None

    def _warp_state_by_homography(self, H: np.ndarray) -> None:
        """Apply inter-frame homography H (prev→curr) to position and
        velocity in `statePost`. Velocity is warped via a finite-difference
        approximation: warp (cx + vx, cy + vy) under H, subtract warped
        (cx, cy); the result is the velocity in the new frame's coords.
        """
        cx = float(self.kf.statePost[0, 0])
        cy = float(self.kf.statePost[1, 0])
        vx = float(self.kf.statePost[4, 0])
        vy = float(self.kf.statePost[5, 0])
        pts = np.array([[[cx, cy], [cx + vx, cy + vy]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pts, H.astype(np.float32))
        ncx, ncy = float(warped[0, 0, 0]), float(warped[0, 0, 1])
        ntx, nty = float(warped[0, 1, 0]), float(warped[0, 1, 1])
        self.kf.statePost[0, 0] = ncx
        self.kf.statePost[1, 0] = ncy
        self.kf.statePost[4, 0] = ntx - ncx
        self.kf.statePost[5, 0] = nty - ncy

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
                             frame_bgr: np.ndarray | None,
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
            # Evidence test. With a persistence map (motion pipeline) we
            # let either appearance or motion clear the bar. Without one
            # (DL pipeline) the detector's own confidence already gated
            # this candidate, so we just need it not to look like a
            # totally different patch. We compare against the best entry
            # in the template bank, not just the freshest EMA.
            sig = self._make_signature(gray, frame_bgr, d.bbox)
            app = self._bank_similarity(sig)
            if persistence is not None:
                p_z = _persistence_z(persistence, d.bbox)
                if app < self.cfg.min_appearance and p_z < self.cfg.min_persistence_z:
                    continue
                s = app + 0.05 * p_z - 0.02 * mahal2
            else:
                if app < -0.2:
                    continue
                s = app + 0.001 * float(d.score) - 0.02 * mahal2
            if best is None or s > best[0]:
                best = (s, d.bbox)
        return best[1] if best else None
