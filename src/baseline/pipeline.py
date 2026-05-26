"""End-to-end baseline pipeline: detect on frame 1, track thereafter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .detector import Detection, DetectorConfig, detect
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
