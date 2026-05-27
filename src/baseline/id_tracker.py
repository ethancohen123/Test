"""Multi-object identity tracker (BoT-SORT-style).

What this gives us on top of the single-target trackers in
`tracker.py`:

- A *list* of tracks, each with a persistent integer ``id`` that
  survives across frames.
- Three-stage data association:
    Stage 1 — IoU matching on high-confidence detections;
    Stage 2 — appearance (HOG cosine) on the remainder;
    Stage 3 — re-identification of unmatched detections against a
              ``lost`` pool of tracks that recently dropped from
              active. Same id is reissued when appearance matches.
- Birth (Stage 4) for anything still unmatched.
- Death: an unmatched active track coasts for ``max_coast_frames`` and
  then moves to the lost pool, where it lives for at most
  ``max_lost_age`` more frames before being expired.

The Kalman filter, ReID extractor, camera-motion warp, and HOG bank
all reuse the same building blocks the single-target tracker already
uses (`tracker._make_kalman`, `tracker._bbox_to_meas`, …,
`reid.make_reid_extractor`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import cv2
import numpy as np

from .detector import Detection
from .tracker import (TrackState, _bbox_to_meas, _make_kalman,
                       _persistence_z, _state_to_bbox)


# --------------------------------------------------------------------------- #
# Config + per-track state
# --------------------------------------------------------------------------- #


@dataclass
class IdentityTrackerConfig:
    # Two-stage detection split (ByteTrack-style)
    high_conf_thresh: float = 0.15

    # Gating thresholds for the three matching stages
    iou_gate: float = 0.20          # min IoU to call a Stage-1 match
    app_gate: float = 0.30          # min cosine sim for Stage-2 match
    reid_gate: float = 0.45         # min cosine sim for Stage-3 re-ID

    # DeepSORT-style two-state lifecycle.
    #   Tentative tracks (hits < tentative_hits) live a *short* coast
    #   budget and are never rendered publicly. This kills one-shot
    #   YOLO false positives before they pollute the visualisation.
    #   Once a track reaches `tentative_hits`, it is "confirmed" and
    #   gets a much longer coast budget so it survives YOLO misses
    #   (the person hiding behind a bush, fast camera pan, …).
    tentative_hits: int = 3
    tentative_max_coast: int = 3
    confirmed_max_coast: int = 60   # 2 s @ 30 fps

    # How long a lost (formerly confirmed) track can wait in the lost
    # pool before being expired forever.
    max_lost_age: int = 240         # 8 s @ 30 fps

    # Only spawn new tracks from confident detections; below this floor,
    # an unmatched detection is treated as a one-shot noise blob and
    # ignored (no new ID). Calibrated against the actual confidence
    # range of the thermal-trained YOLO checkpoint on our clip (0.05-0.35).
    birth_min_conf: float = 0.10

    reid_bank_size: int = 8
    reid_bank_min_gap: float = 0.985

    # Crop sanity
    min_bbox_side: int = 4


@dataclass
class IdentityTrack:
    """One persistent ID across frames."""
    id: int
    bbox: tuple[int, int, int, int]
    state: TrackState
    coast_frames: int = 0
    age: int = 0
    hits: int = 1                          # frames matched to a detection
    last_seen_frame: int = 0
    last_conf: float = 0.0
    confirmed: bool = False                # promoted from tentative
    appearance: np.ndarray | None = field(default=None, repr=False)
    bank: list[np.ndarray] = field(default_factory=list, repr=False)
    kf: cv2.KalmanFilter = field(default=None, repr=False)  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Geometric / appearance helpers
# --------------------------------------------------------------------------- #


def _iou(a: tuple[int, int, int, int],
         b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx); iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
    if ix2 <= ix or iy2 <= iy:
        return 0.0
    inter = (ix2 - ix) * (iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def _greedy_assign(cost: np.ndarray, max_cost: float
                    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy assignment on a cost matrix `cost` (rows × cols).
    Returns (matches, unmatched_rows, unmatched_cols)."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    flat = []
    nr, nc = cost.shape
    for r in range(nr):
        for c in range(nc):
            cval = float(cost[r, c])
            if cval <= max_cost:
                flat.append((cval, r, c))
    flat.sort()
    matches: list[tuple[int, int]] = []
    used_r: set[int] = set()
    used_c: set[int] = set()
    for _, r, c in flat:
        if r in used_r or c in used_c:
            continue
        matches.append((r, c))
        used_r.add(r); used_c.add(c)
    unmatched_r = [r for r in range(nr) if r not in used_r]
    unmatched_c = [c for c in range(nc) if c not in used_c]
    return matches, unmatched_r, unmatched_c


def _warp_kalman_by_H(kf: cv2.KalmanFilter, H: np.ndarray) -> None:
    """Same camera-motion warp `tracker.Tracker._warp_state_by_homography`
    applies to the single-target tracker."""
    cx = float(kf.statePost[0, 0])
    cy = float(kf.statePost[1, 0])
    vx = float(kf.statePost[4, 0])
    vy = float(kf.statePost[5, 0])
    pts = np.array([[[cx, cy], [cx + vx, cy + vy]]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pts, H.astype(np.float32))
    ncx, ncy = float(warped[0, 0, 0]), float(warped[0, 0, 1])
    ntx, nty = float(warped[0, 1, 0]), float(warped[0, 1, 1])
    kf.statePost[0, 0] = ncx
    kf.statePost[1, 0] = ncy
    kf.statePost[4, 0] = ntx - ncx
    kf.statePost[5, 0] = nty - ncy


def _sanitise_bbox(bbox: tuple[int, int, int, int],
                    shape: tuple[int, int],
                    min_side: int = 4
                    ) -> tuple[int, int, int, int] | None:
    H, W = shape
    x, y, w, h = (int(v) for v in bbox)
    x = max(0, min(x, W - min_side))
    y = max(0, min(y, H - min_side))
    w = max(min_side, min(w, W - x))
    h = max(min_side, min(h, H - y))
    if w < min_side or h < min_side:
        return None
    return x, y, w, h


# --------------------------------------------------------------------------- #
# The multi-track manager
# --------------------------------------------------------------------------- #


class IdentityTracker:
    """List-of-tracks manager. Same Kalman + HOG building blocks as
    the single-target tracker, plumbed for many targets at once.
    """

    def __init__(self, cfg: IdentityTrackerConfig | None = None):
        from .reid import make_reid_extractor
        self.cfg = cfg or IdentityTrackerConfig()
        self.active: list[IdentityTrack] = []
        self.lost: list[IdentityTrack] = []
        self.next_id: int = 1
        self._frame_idx: int = 0
        self._reid = make_reid_extractor()

    # ----- public reset (e.g., on modality switch) -----
    def reset(self) -> None:
        """Drop every track (lost included). Used by the pipeline on a
        modality switch — the new regime has no continuity with the
        previous one's identities."""
        self.active = []
        self.lost = []
        # we deliberately keep next_id so new IDs after a reset don't
        # collide with old ones in the same output video

    # ----- helpers used by step() -----
    def _embed(self, frame_bgr: np.ndarray,
               bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        return self._reid.embed(frame_bgr, bbox) if self._reid is not None else None

    @staticmethod
    def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
        if a is None or b is None or a.shape != b.shape:
            return 0.0
        return float(np.dot(a, b))

    def _bank_best(self, sig: np.ndarray | None,
                   bank: list[np.ndarray]) -> float:
        if sig is None or not bank:
            return 0.0
        return max(self._cosine(sig, t) for t in bank)

    def _push_bank(self, track: IdentityTrack, sig: np.ndarray | None) -> None:
        if sig is None:
            return
        if track.bank and self._cosine(sig, track.bank[0]) >= self.cfg.reid_bank_min_gap:
            return
        track.bank.insert(0, sig)
        if len(track.bank) > self.cfg.reid_bank_size:
            track.bank = track.bank[: self.cfg.reid_bank_size]

    def _new_kalman_from_bbox(self, bbox: tuple[int, int, int, int]
                              ) -> cv2.KalmanFilter:
        kf = _make_kalman()
        meas = _bbox_to_meas(bbox)
        kf.statePost = np.vstack([meas, np.zeros((2, 1), dtype=np.float32)])
        kf.statePre = kf.statePost.copy()
        return kf

    def _update_track(self, t: IdentityTrack, d: Detection,
                       gray: np.ndarray, frame_bgr: np.ndarray) -> None:
        sane = _sanitise_bbox(d.bbox, gray.shape, self.cfg.min_bbox_side)
        if sane is None:
            return
        t.kf.correct(_bbox_to_meas(sane))
        t.bbox = _state_to_bbox(t.kf.statePost[:4])
        t.state = TrackState.TRACKING
        t.coast_frames = 0
        t.hits += 1
        t.last_seen_frame = self._frame_idx
        t.last_conf = float(d.score)
        if t.hits >= self.cfg.tentative_hits:
            t.confirmed = True
        sig = self._embed(frame_bgr, sane)
        if sig is not None:
            t.appearance = sig
            self._push_bank(t, sig)

    def _birth_track(self, d: Detection,
                      gray: np.ndarray, frame_bgr: np.ndarray) -> None:
        sane = _sanitise_bbox(d.bbox, gray.shape, self.cfg.min_bbox_side)
        if sane is None:
            return
        kf = self._new_kalman_from_bbox(sane)
        sig = self._embed(frame_bgr, sane)
        track = IdentityTrack(
            id=self.next_id,
            bbox=sane,
            state=TrackState.TRACKING,
            coast_frames=0,
            age=1,
            hits=1,
            last_seen_frame=self._frame_idx,
            last_conf=float(d.score),
            appearance=sig,
            bank=[sig] if sig is not None else [],
            kf=kf,
        )
        self.next_id += 1
        self.active.append(track)

    def _reactivate(self, t: IdentityTrack, d: Detection,
                     gray: np.ndarray, frame_bgr: np.ndarray) -> None:
        """Re-bind a track from the lost pool to a fresh detection."""
        sane = _sanitise_bbox(d.bbox, gray.shape, self.cfg.min_bbox_side)
        if sane is None:
            return
        t.kf = self._new_kalman_from_bbox(sane)
        t.bbox = sane
        t.state = TrackState.TRACKING
        t.coast_frames = 0
        t.hits += 1
        t.last_seen_frame = self._frame_idx
        t.last_conf = float(d.score)
        sig = self._embed(frame_bgr, sane)
        if sig is not None:
            t.appearance = sig
            self._push_bank(t, sig)

    # ----- the main per-frame loop -----
    def step(self,
             detections: list[Detection],
             gray: np.ndarray,
             frame_bgr: np.ndarray,
             ego_motion_H: np.ndarray | None = None,
             ) -> list[IdentityTrack]:
        self._frame_idx += 1

        # 1) Predict every active track (camera-motion-aware).
        for t in self.active:
            if ego_motion_H is not None:
                _warp_kalman_by_H(t.kf, ego_motion_H)
            t.kf.predict()
            pred_bbox = _state_to_bbox(t.kf.statePre[:4])
            # Sanitise prediction to image (so subsequent IoU is sane).
            sane_pred = _sanitise_bbox(pred_bbox, gray.shape, self.cfg.min_bbox_side)
            if sane_pred is not None:
                t.bbox = sane_pred  # tentative pre-match bbox
            t.age += 1

        # 2) Split detections by confidence (ByteTrack two-stage idea).
        high_idx = [i for i, d in enumerate(detections)
                     if d.score >= self.cfg.high_conf_thresh]
        low_idx = [i for i, d in enumerate(detections)
                    if d.score < self.cfg.high_conf_thresh]

        unmatched_active = list(range(len(self.active)))
        unmatched_dets = set(range(len(detections)))

        # ---- Stage 1: IoU matching with HIGH-confidence detections ----
        if unmatched_active and high_idx:
            cost = np.ones((len(unmatched_active), len(high_idx)),
                            dtype=np.float32)
            for ri, ai in enumerate(unmatched_active):
                for ci, di in enumerate(high_idx):
                    iou = _iou(self.active[ai].bbox, detections[di].bbox)
                    cost[ri, ci] = 1.0 - iou
            matches, ur, uc = _greedy_assign(cost, 1.0 - self.cfg.iou_gate)
            for ri, ci in matches:
                ai = unmatched_active[ri]; di = high_idx[ci]
                self._update_track(self.active[ai], detections[di],
                                    gray, frame_bgr)
                unmatched_dets.discard(di)
            unmatched_active = [unmatched_active[r] for r in ur]

        # ---- Stage 2: APPEARANCE matching for the rest ----
        # Pool = unmatched high-conf  ∪  all low-conf
        stage2_det_idx = [di for di in high_idx if di in unmatched_dets] + low_idx
        if unmatched_active and stage2_det_idx:
            cost = np.ones((len(unmatched_active), len(stage2_det_idx)),
                            dtype=np.float32)
            embeddings = [self._embed(frame_bgr, detections[di].bbox)
                           for di in stage2_det_idx]
            for ri, ai in enumerate(unmatched_active):
                tbank = self.active[ai].bank
                for ci, _di in enumerate(stage2_det_idx):
                    sim = self._bank_best(embeddings[ci], tbank)
                    cost[ri, ci] = 1.0 - sim
            matches, ur, uc = _greedy_assign(cost, 1.0 - self.cfg.app_gate)
            for ri, ci in matches:
                ai = unmatched_active[ri]; di = stage2_det_idx[ci]
                self._update_track(self.active[ai], detections[di],
                                    gray, frame_bgr)
                unmatched_dets.discard(di)
            unmatched_active = [unmatched_active[r] for r in ur]

        # ---- Stage 3: re-ID against the LOST pool ----
        if self.lost and unmatched_dets:
            still_idx = sorted(unmatched_dets)
            cost = np.ones((len(self.lost), len(still_idx)), dtype=np.float32)
            embeddings = [self._embed(frame_bgr, detections[di].bbox)
                           for di in still_idx]
            for ri, t in enumerate(self.lost):
                for ci, _di in enumerate(still_idx):
                    sim = self._bank_best(embeddings[ci], t.bank)
                    cost[ri, ci] = 1.0 - sim
            matches, _, _ = _greedy_assign(cost, 1.0 - self.cfg.reid_gate)
            reactivated_lost_rows: set[int] = set()
            for ri, ci in matches:
                t = self.lost[ri]
                di = still_idx[ci]
                self._reactivate(t, detections[di], gray, frame_bgr)
                unmatched_dets.discard(di)
                reactivated_lost_rows.add(ri)
            # Move reactivated lost tracks back to active.
            new_lost = []
            for i, t in enumerate(self.lost):
                if i in reactivated_lost_rows:
                    self.active.append(t)
                else:
                    new_lost.append(t)
            self.lost = new_lost

        # ---- Stage 4: birth new tracks for the remaining detections ----
        # Only confident detections seed an ID — otherwise every YOLO
        # false positive would spawn its own track and clutter the
        # visualisation. The new track is *tentative* until it has
        # accumulated `tentative_hits` matched frames.
        for di in sorted(unmatched_dets):
            d = detections[di]
            if d.score >= self.cfg.birth_min_conf:
                self._birth_track(d, gray, frame_bgr)

        # ---- Coast every active track that didn't get matched ----
        # Two budgets, DeepSORT-style: tentative tracks die fast,
        # confirmed tracks survive a long YOLO silence.
        survived_active: list[IdentityTrack] = []
        for t in self.active:
            if t.last_seen_frame == self._frame_idx:
                survived_active.append(t)
                continue
            t.coast_frames += 1
            t.bbox = _state_to_bbox(t.kf.statePost[:4])
            sane = _sanitise_bbox(t.bbox, gray.shape, self.cfg.min_bbox_side)
            if sane is not None:
                t.bbox = sane

            is_confirmed = t.hits >= self.cfg.tentative_hits
            budget = (self.cfg.confirmed_max_coast if is_confirmed
                      else self.cfg.tentative_max_coast)
            if t.coast_frames > budget:
                if is_confirmed:
                    # Confirmed tracks go to the lost pool for ReID.
                    t.state = TrackState.LOST
                    self.lost.append(t)
                # Tentative tracks just die quietly — no ReID, no slot
                # in the lost pool. (They were probably noise.)
            else:
                t.state = TrackState.COASTING
                survived_active.append(t)
        self.active = survived_active

        # ---- Expire old lost tracks ----
        self.lost = [
            t for t in self.lost
            if (self._frame_idx - t.last_seen_frame) <= self.cfg.max_lost_age
        ]

        return self.active
