"""v2 motion + DL verifier + 1-shot exemplar tracker leg.

Wraps :class:`baseline_v2_hybrid.HybridV2Pipeline` (which is itself a
strict superset of v2-motion) and adds an exemplar-seeded CSRT
tracker as a third source of evidence:

    motion  (v2)        — change detection in the scene
    DL      (YOLO)      — class evidence (every K frames)
    exemplar (CSRT)     — *instance* evidence locked to a 1-shot template

The exemplar tracker is the new piece. It is seeded once, either from
an explicit (frame, bbox) — typically a CVAT-annotated human-sized
crop, the "1-shot" support — or, if no exemplar is provided, from the
first frame where v2-hybrid emits a track that DL marks VERIFIED.

Fusion rule is intentionally simple so we can read what improved:

    * If the exemplar tracker is initialised and reports OK, its bbox
      is the headline output.
    * If the tracker disagrees with v2-hybrid (IoU < lost_iou_thresh)
      for too many consecutive frames, we treat it as drift and reset.
    * If the tracker is uninitialised (or lost), we fall back to
      whatever v2-hybrid would have returned. v2 and v2-hybrid are
      not modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np

from baseline_v2.tracker import TrackState, TrackState_
from baseline_v2_hybrid.pipeline import (DLStatus, HybridStepResult,
                                          HybridV2Config, HybridV2Pipeline)

from .exemplar_tracker import ExemplarTracker

BBox = Tuple[int, int, int, int]


# --------------------------------------------------------------------------- #
# Config / result
# --------------------------------------------------------------------------- #


@dataclass
class HybridV3Config:
    base: HybridV2Config = field(default_factory=HybridV2Config)

    # Optional manual seeding. If both fields are set, the exemplar
    # tracker is initialised from this (frame, bbox) the moment the
    # pipeline sees ``frame``. Leave at None to auto-seed from the first
    # DL-VERIFIED frame.
    exemplar_frame: Optional[int] = None
    exemplar_bbox: Optional[BBox] = None

    # When the tracker disagrees with v2-hybrid by less than this IoU
    # for ``consec_disagree_before_reset`` consecutive frames, we
    # consider the tracker to have drifted and reset it. v2-hybrid
    # then re-takes the wheel.
    lost_iou_thresh: float = 0.05
    consec_disagree_before_reset: int = 8

    # Once reset, may we re-seed the tracker from a later DL-VERIFIED
    # frame? Almost always yes.
    reinit_on_verified: bool = True


@dataclass
class HybridV3StepResult:
    """Mirrors ``HybridStepResult`` so the existing v2 visualiser
    keeps working unchanged; adds the v3-specific fields at the end."""
    # ---- v2-hybrid pass-through ----
    frame_idx: int
    gray: np.ndarray
    candidates: list
    track: Optional[TrackState_]
    persistence: Optional[np.ndarray]
    modality_switched: bool
    dl_status: DLStatus
    dl_candidates: list
    dl_was_run: bool
    reseeded: bool
    # ---- v3 additions ----
    tracker_bbox: Optional[BBox]
    tracker_source: str          # "exemplar" | "v2_hybrid" | "none"
    tracker_initialised: bool
    tracker_just_seeded: bool


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx); iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
    if ix2 <= ix or iy2 <= iy:
        return 0.0
    inter = (ix2 - ix) * (iy2 - iy)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def _synth_trackstate(bbox: BBox, score: float = 1.0) -> TrackState_:
    """Build a TrackState_ that the existing v2 visualiser can render."""
    return TrackState_(bbox=bbox, state=TrackState.TRACKING,
                       coast_frames=0, score=float(score))


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class HybridV3Pipeline:
    def __init__(self, cfg: Optional[HybridV3Config] = None) -> None:
        self.cfg = cfg or HybridV3Config()
        self.base = HybridV2Pipeline(self.cfg.base)
        self.tracker = ExemplarTracker()
        self._consec_disagree = 0

    # ---- per-frame ----
    def step(self, frame_idx: int, frame_bgr: np.ndarray) -> HybridV3StepResult:
        # 1. v2-hybrid runs unchanged.
        base_res: HybridStepResult = self.base.step(frame_idx, frame_bgr)
        v2_bbox = base_res.track.bbox if base_res.track is not None else None

        just_seeded = False
        tk_bbox: Optional[BBox] = None
        tk_source = "none"

        # 2. Seed tracker on the exemplar frame if configured.
        if (not self.tracker.initialised
                and self.cfg.exemplar_frame is not None
                and self.cfg.exemplar_bbox is not None
                and frame_idx == self.cfg.exemplar_frame):
            self.tracker.init(frame_bgr, self.cfg.exemplar_bbox,
                              frame_idx=frame_idx)
            tk_bbox = self.cfg.exemplar_bbox
            tk_source = "exemplar"
            just_seeded = True

        # 3. Otherwise auto-seed from first DL-VERIFIED frame.
        elif (not self.tracker.initialised
              and self.cfg.exemplar_frame is None
              and v2_bbox is not None
              and base_res.dl_status == DLStatus.VERIFIED):
            self.tracker.init(frame_bgr, v2_bbox, frame_idx=frame_idx)
            tk_bbox = v2_bbox
            tk_source = "exemplar"
            just_seeded = True

        # 4. Running tracker. update() once per frame after seeding.
        elif self.tracker.initialised:
            ok, bbox = self.tracker.update(frame_bgr)
            if ok and bbox is not None:
                tk_bbox = bbox
                tk_source = "exemplar"
                # Drift check against v2-hybrid (only if v2 has a track).
                if v2_bbox is not None:
                    if _iou(bbox, v2_bbox) < self.cfg.lost_iou_thresh:
                        self._consec_disagree += 1
                        if (self._consec_disagree
                                >= self.cfg.consec_disagree_before_reset):
                            self.tracker.reset()
                            self._consec_disagree = 0
                            tk_bbox = v2_bbox
                            tk_source = "v2_hybrid"
                    else:
                        self._consec_disagree = 0
            else:
                # CSRT lost lock.
                self.tracker.reset()
                self._consec_disagree = 0
                tk_bbox = v2_bbox
                tk_source = "v2_hybrid" if v2_bbox is not None else "none"
        else:
            # Tracker still not seeded — pass through v2-hybrid.
            tk_bbox = v2_bbox
            tk_source = "v2_hybrid" if v2_bbox is not None else "none"

        # 5. Build a result that *looks like* a v2-hybrid result so the
        #    existing visualisers and the eval script work unchanged.
        if tk_bbox is not None and tk_source == "exemplar":
            # Override the track field with the exemplar bbox so the
            # downstream eval ("pred = res.track.bbox") returns the
            # exemplar-tracker bbox, not the original v2 one.
            track_field = _synth_trackstate(tk_bbox)
        else:
            track_field = base_res.track

        return HybridV3StepResult(
            frame_idx=base_res.frame_idx,
            gray=base_res.gray,
            candidates=base_res.candidates,
            track=track_field,
            persistence=base_res.persistence,
            modality_switched=base_res.modality_switched,
            dl_status=base_res.dl_status,
            dl_candidates=base_res.dl_candidates,
            dl_was_run=base_res.dl_was_run,
            reseeded=base_res.reseeded,
            tracker_bbox=tk_bbox,
            tracker_source=tk_source,
            tracker_initialised=self.tracker.initialised,
            tracker_just_seeded=just_seeded,
        )

    def run(self, frames: Iterable[tuple[int, np.ndarray]]):
        for idx, bgr in frames:
            yield self.step(idx, bgr)
