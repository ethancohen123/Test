"""1-shot exemplar-seeded CSRT tracker.

A thin wrapper around ``cv2.TrackerCSRT`` whose only purpose is to make
the *initialisation* explicit. Unlike the per-frame CSRT inside
``baseline_v2.tracker`` — which gets re-initialised from whatever
small motion blob the v2 detector found — this tracker is seeded
once from a single exemplar bounding box (the "1-shot" template,
typically a CVAT-annotated human-sized crop) and keeps tracking
until it loses lock.

That difference matters because CSRT propagates a box at
approximately the size it was initialised with. Seeding from a
human-sized template therefore produces human-sized predicted
boxes, which directly addresses the size mismatch we observed
when v2's CSRT was seeded from a ~30 px bright blob against a
~110 px ground-truth box.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


def _make_csrt() -> "cv2.Tracker":
    """OpenCV 4 moved CSRT in/out of cv2.legacy a couple of times.
    Try the modern path first, fall back to legacy."""
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    raise RuntimeError("OpenCV build lacks TrackerCSRT")


class ExemplarTracker:
    """Stateful 1-shot tracker."""

    def __init__(self) -> None:
        self._csrt: Optional[cv2.Tracker] = None
        self._init_bbox: Optional[BBox] = None
        self._init_frame: Optional[int] = None
        self._frames_since_init: int = 0

    @property
    def initialised(self) -> bool:
        return self._csrt is not None

    @property
    def init_bbox(self) -> Optional[BBox]:
        return self._init_bbox

    @property
    def frames_since_init(self) -> int:
        return self._frames_since_init

    def init(self, frame: np.ndarray, bbox: BBox,
             frame_idx: Optional[int] = None) -> None:
        """Seed the tracker from a frame + bbox. Re-callable for resets."""
        self._csrt = _make_csrt()
        # CSRT wants ints
        x, y, w, h = (int(v) for v in bbox)
        self._csrt.init(frame, (x, y, w, h))
        self._init_bbox = (x, y, w, h)
        self._init_frame = frame_idx
        self._frames_since_init = 0

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[BBox]]:
        """Returns (ok, bbox). ``ok=False`` means CSRT lost lock; the
        caller should typically reset and fall back to detection."""
        if self._csrt is None:
            return False, None
        ok, raw = self._csrt.update(frame)
        self._frames_since_init += 1
        if not ok:
            return False, None
        x, y, w, h = raw
        return True, (int(x), int(y), int(w), int(h))

    def reset(self) -> None:
        self._csrt = None
        self._init_bbox = None
        self._init_frame = None
        self._frames_since_init = 0
