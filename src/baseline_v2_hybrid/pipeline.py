"""v2-motion-pipeline + DL verifier."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Iterable

import numpy as np

# v2 baseline (unchanged) ----------------------------------------------------
from baseline_v2.detector import Detection as V2Detection
from baseline_v2.pipeline import MotionPipeline as V2MotionPipeline
from baseline_v2.pipeline import MotionPipelineConfig as V2MotionPipelineConfig
from baseline_v2.pipeline import StepResult as V2StepResult
from baseline_v2.tracker import TrackState as V2TrackState
from baseline_v2.preprocess import preprocess as v2_preprocess
from baseline_v2.preprocess import to_gray as v2_to_gray

# DL detector (current tree) -------------------------------------------------
# We only need YOLO inference; nothing about the post-v2 tracker / hybrid is
# imported here.
from baseline.dl_detector import DLDetector, DLDetectorConfig


# --------------------------------------------------------------------------- #
# DL status enum + per-frame result bundle
# --------------------------------------------------------------------------- #


class DLStatus(Enum):
    SILENT = auto()        # DL fired nothing this check
    VERIFIED = auto()      # DL overlapped v2's bbox (IoU >= threshold)
    CONTRADICTED = auto()  # DL fired but did not overlap v2's bbox
    INACTIVE = auto()      # DL not run this frame (between K-frame ticks)
    SEEDLESS = auto()      # v2 has no track AND DL didn't see anything


@dataclass
class HybridStepResult:
    """What the runner gets per frame.

    Mirrors the v2 StepResult shape (so the v2 visualiser can render it
    unchanged), plus the new DL fields.
    """
    frame_idx: int
    gray: np.ndarray
    candidates: list  # v2 candidates (motion blobs)
    track: object | None  # v2 TrackState_
    persistence: np.ndarray | None
    modality_switched: bool

    dl_status: DLStatus
    dl_candidates: list                # DL detections (this frame, if run)
    dl_was_run: bool                    # True only on every-K-frame ticks
    reseeded: bool                      # True on the frame we forced a reseed


# --------------------------------------------------------------------------- #
# Config + helpers
# --------------------------------------------------------------------------- #


@dataclass
class HybridV2Config:
    motion: V2MotionPipelineConfig = field(default_factory=V2MotionPipelineConfig)
    dl: DLDetectorConfig | None = None

    # How often to call YOLO (frames). K=3 ≈ 10 Hz on a 30 fps source.
    dl_every_k: int = 3

    # A DL detection "agrees with" the v2 bbox if their IoU is >= this.
    dl_iou_thresh: float = 0.20

    # Only DL detections above this confidence participate (both for
    # verification and for reseeding).
    dl_min_conf: float = 0.10

    # We require this many *consecutive* CONTRADICTED checks before
    # reseeding v2 from the DL detection. Prevents one-frame YOLO false
    # positives from hijacking the v2 track.
    contradicted_before_reseed: int = 3


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx); iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
    if ix2 <= ix or iy2 <= iy:
        return 0.0
    inter = (ix2 - ix) * (iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


# --------------------------------------------------------------------------- #
# The wrapper pipeline
# --------------------------------------------------------------------------- #


class HybridV2Pipeline:
    """v2 motion + DL verifier (every K frames)."""

    def __init__(self, cfg: HybridV2Config | None = None):
        self.cfg = cfg or HybridV2Config()
        self.v2 = V2MotionPipeline(self.cfg.motion)
        self.dl = DLDetector(self.cfg.dl or DLDetectorConfig())

        # state
        self._consec_contradicted: int = 0
        self._last_dl_status: DLStatus = DLStatus.INACTIVE
        self._last_dl_cands: list = []

    # ----- the per-frame loop -----
    def step(self, frame_idx: int, frame_bgr: np.ndarray) -> HybridStepResult:
        # 1) v2 runs unchanged.
        v2_res: V2StepResult = self.v2.step(frame_idx, frame_bgr)

        # 2) Decide whether DL runs this frame.
        run_dl = (frame_idx % self.cfg.dl_every_k == 0)
        dl_status = DLStatus.INACTIVE
        dl_cands: list = []
        reseeded = False

        if run_dl:
            dl_cands = self.dl(frame_bgr, invert=False)
            dl_cands = [d for d in dl_cands if d.score >= self.cfg.dl_min_conf]
            self._last_dl_cands = dl_cands

            dl_status = self._classify_dl(v2_res, dl_cands)
            self._last_dl_status = dl_status

            if dl_status == DLStatus.CONTRADICTED:
                self._consec_contradicted += 1
            else:
                self._consec_contradicted = 0

            # 3) Reseed v2 only after N consecutive CONTRADICTED ticks.
            if self._consec_contradicted >= self.cfg.contradicted_before_reseed:
                best_dl = max(dl_cands, key=lambda d: d.score, default=None)
                if best_dl is not None:
                    self._reseed_v2(best_dl, frame_bgr, v2_res.gray)
                    reseeded = True
                    self._consec_contradicted = 0
                    # re-fetch v2's state after reseed so the bbox the
                    # runner draws reflects the new identity
                    v2_res = self._latest_v2_state(v2_res, best_dl)
        else:
            # Carry forward the last DL status so the HUD doesn't blink
            # between ticks.
            dl_status = self._last_dl_status
            dl_cands = self._last_dl_cands

        return HybridStepResult(
            frame_idx=frame_idx,
            gray=v2_res.gray,
            candidates=v2_res.candidates,
            track=v2_res.track,
            persistence=v2_res.persistence,
            modality_switched=v2_res.modality_switched,
            dl_status=dl_status,
            dl_candidates=dl_cands,
            dl_was_run=run_dl,
            reseeded=reseeded,
        )

    def run(self, frames: Iterable[tuple[int, np.ndarray]]) -> Iterable[HybridStepResult]:
        for idx, bgr in frames:
            yield self.step(idx, bgr)

    # ----- internals -----
    def _classify_dl(self, v2_res: V2StepResult, dl_cands: list) -> DLStatus:
        v2_bbox = v2_res.track.bbox if v2_res.track is not None else None
        if not dl_cands:
            # DL fired nothing. We can't say anything.
            return DLStatus.SILENT
        if v2_bbox is None:
            # v2 has no track but DL sees something. That's not strictly
            # a contradiction (v2 hasn't claimed anything yet), but for
            # our purpose it should accumulate so we eventually reseed
            # if v2 stays empty.
            return DLStatus.SEEDLESS
        max_iou = max(_iou(v2_bbox, d.bbox) for d in dl_cands)
        if max_iou >= self.cfg.dl_iou_thresh:
            return DLStatus.VERIFIED
        return DLStatus.CONTRADICTED

    def _reseed_v2(self, det, frame_bgr: np.ndarray,
                    v2_gray: np.ndarray) -> None:
        """Force v2's tracker onto a DL detection. We reach into v2's
        Tracker.init using v2's own Detection type to keep the seed
        consistent with whatever v2's tracker expects.
        """
        v2_det = V2Detection(
            bbox=det.bbox,
            score=float(det.score),
            area=int(det.area),
        )
        # v2's tracker has its own init signature (no frame_bgr arg).
        self.v2.tracker.init(v2_gray, v2_det)
        self.v2._initialised = True

    def _latest_v2_state(self, prev_v2_res: V2StepResult, seed_det) -> V2StepResult:
        """After a reseed, the v2 step we already ran is now stale; we
        synthesize a fresh-looking StepResult with the seed bbox so the
        runner's per-frame visualisation reflects what the next frame
        will start from."""
        from baseline_v2.tracker import TrackState_ as V2TS_
        ts = V2TS_(
            bbox=seed_det.bbox,
            state=V2TrackState.TRACKING,
            coast_frames=0,
            score=float(seed_det.score),
        )
        return V2StepResult(
            frame_idx=prev_v2_res.frame_idx,
            gray=prev_v2_res.gray,
            candidates=prev_v2_res.candidates,
            track=ts,
            persistence=prev_v2_res.persistence,
            modality_switched=prev_v2_res.modality_switched,
        )
