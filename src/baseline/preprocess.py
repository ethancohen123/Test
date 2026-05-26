"""Thermal-aware preprocessing: grayscale, polarity, CLAHE, denoise."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocConfig:
    clahe_clip: float = 2.5
    clahe_grid: int = 8
    blur_ksize: int = 3            # 0 to disable
    assume_white_hot: bool | None = None  # None = auto-detect on the first frame


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.ndim == 2:
        return frame_bgr
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def detect_polarity_white_hot(gray: np.ndarray) -> bool:
    """Heuristic: thermal targets are *rare* bright (or dark) blobs.
    If the right tail of the histogram is heavier than the left relative
    to the median, scene is white-hot. Used only for the first frame; the
    result is then frozen for the clip.
    """
    med = np.median(gray)
    upper = float(np.mean(gray > med + 25))
    lower = float(np.mean(gray < med - 25))
    return upper >= lower


def apply_clahe(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def preprocess(frame_bgr: np.ndarray, cfg: PreprocConfig,
               white_hot: bool) -> np.ndarray:
    """Return an 8-bit single-channel image where the target is bright."""
    gray = to_gray(frame_bgr)
    if not white_hot:
        gray = cv2.bitwise_not(gray)
    gray = apply_clahe(gray, cfg.clahe_clip, cfg.clahe_grid)
    if cfg.blur_ksize and cfg.blur_ksize >= 3:
        gray = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
    return gray
