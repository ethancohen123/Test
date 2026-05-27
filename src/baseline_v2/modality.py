"""Detect the thermal ↔ infrared switch the clip contains.

We don't try to classify the *kind* of modality — we just detect a hard
step in the per-frame intensity statistics and emit a 'switched' event.
The pipeline uses that event to reset its temporal state (persistence
map, polarity, tracker) so stale state from before the switch does not
poison the new regime.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ModalityConfig:
    hist_bins: int = 32
    chi2_threshold: float = 1.5     # chi-square distance between hists
    cooldown_frames: int = 10        # ignore further switches for N frames
    color_channel_threshold: float = 8.0  # |R-B| mean diff to call it colour


class ModalityMonitor:
    def __init__(self, cfg: ModalityConfig | None = None):
        self.cfg = cfg or ModalityConfig()
        self._prev_hist: np.ndarray | None = None
        self._cooldown: int = 0
        self._last_distance: float = 0.0
        self._last_is_color: bool = False

    @staticmethod
    def _hist(gray: np.ndarray, bins: int) -> np.ndarray:
        h = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten()
        s = h.sum()
        return h / s if s > 0 else h

    @staticmethod
    def _chi2(a: np.ndarray, b: np.ndarray) -> float:
        eps = 1e-6
        return float(0.5 * ((a - b) ** 2 / (a + b + eps)).sum())

    def is_color_frame(self, frame_bgr: np.ndarray) -> bool:
        if frame_bgr.ndim != 3:
            return False
        b = frame_bgr[..., 0].astype(np.float32)
        r = frame_bgr[..., 2].astype(np.float32)
        return float(np.abs(r - b).mean()) > self.cfg.color_channel_threshold

    def step(self, gray: np.ndarray, frame_bgr: np.ndarray) -> bool:
        """Return True iff this frame begins a new modality regime."""
        is_color = self.is_color_frame(frame_bgr)
        hist = self._hist(gray, self.cfg.hist_bins)
        switched = False
        if self._prev_hist is not None and self._cooldown == 0:
            d = self._chi2(hist, self._prev_hist)
            self._last_distance = d
            color_flip = is_color != self._last_is_color
            if d > self.cfg.chi2_threshold or color_flip:
                switched = True
                self._cooldown = self.cfg.cooldown_frames
        self._prev_hist = hist
        self._last_is_color = is_color
        if self._cooldown > 0:
            self._cooldown -= 1
        return switched
