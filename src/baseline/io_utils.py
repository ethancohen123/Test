"""Video I/O helpers: frame iteration and annotated writeback."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    n_frames: int


def probe(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    meta = VideoMeta(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def iter_frames(path: str | Path, start: int = 0, stop: int | None = None,
                step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame) pairs."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if stop is not None and idx >= stop:
                return
            if (idx - start) % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


def read_frame(path: str | Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise IndexError(f"Frame {index} out of range for {path}")
        return frame
    finally:
        cap.release()


class VideoWriter:
    """Lazy mp4 writer; opens on first write so we can size from the first frame."""

    def __init__(self, path: str | Path, fps: float, fourcc: str = "mp4v"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self._w: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None

    def write(self, frame: np.ndarray) -> None:
        if self._w is None:
            h, w = frame.shape[:2]
            self._size = (w, h)
            self._w = cv2.VideoWriter(str(self.path), self.fourcc, self.fps, (w, h))
            if not self._w.isOpened():
                raise RuntimeError(f"Could not open writer for {self.path}")
        self._w.write(frame)

    def close(self) -> None:
        if self._w is not None:
            self._w.release()
            self._w = None

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
