"""Ego-motion estimation and motion-compensated temporal persistence.

The drone camera moves, so plain frame-differencing is dominated by
parallax/translation, not by the target's motion. We estimate the
camera's frame-to-frame homography with ORB+RANSAC, warp the previous
frame into the current frame's coordinates, and treat the absolute
difference as a "what moved relative to the ground" signal. We then
accumulate that signal over time (an EMA that is itself warped each
frame) so a persistently moving target stands out from one-shot noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionConfig:
    orb_n_features: int = 800
    ransac_reproj: float = 3.0
    min_inliers: int = 25
    diff_blur_ksize: int = 5      # smooth diff before accumulation
    persistence_alpha: float = 0.6  # EMA weight on history vs new diff
    persistence_decay_floor: float = 0.0


# ---------- ego-motion ----------

def estimate_homography(prev_gray: np.ndarray, curr_gray: np.ndarray,
                        cfg: MotionConfig) -> np.ndarray | None:
    """Return 3x3 homography that maps prev_gray points → curr_gray, or None."""
    orb = cv2.ORB_create(nfeatures=cfg.orb_n_features, fastThreshold=10)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = list(bf.match(des1, des2))
    if len(matches) < cfg.min_inliers:
        return None
    matches.sort(key=lambda m: m.distance)
    matches = matches[: max(80, len(matches) // 2)]

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_reproj)
    if H is None or mask is None or int(mask.sum()) < cfg.min_inliers:
        return None
    return H


def warp_like(img: np.ndarray, H: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Warp `img` into the coordinate frame of `ref` using H (prev→curr)."""
    h, w = ref.shape[:2]
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def compensated_diff(prev_gray: np.ndarray, curr_gray: np.ndarray,
                     H: np.ndarray | None,
                     cfg: MotionConfig) -> np.ndarray:
    """abs(curr - warp(prev, H)); plain abs-diff if H is None."""
    if H is None:
        warped = prev_gray
    else:
        warped = warp_like(prev_gray, H, curr_gray)
    diff = cv2.absdiff(curr_gray, warped)
    if cfg.diff_blur_ksize and cfg.diff_blur_ksize >= 3:
        k = cfg.diff_blur_ksize | 1
        diff = cv2.GaussianBlur(diff, (k, k), 0)
    return diff


# ---------- persistence map ----------

class PersistenceMap:
    """EMA of motion-compensated residuals, kept in image coordinates.

    On each step we (a) warp the stored map into the new frame's
    coordinates, (b) blend in the new diff. Result: a heat-map where
    pixels that consistently move *against* the warped background stay
    hot, while one-shot warp residuals decay away.
    """

    def __init__(self, cfg: MotionConfig):
        self.cfg = cfg
        self._map: np.ndarray | None = None

    def reset(self) -> None:
        self._map = None

    def value(self) -> np.ndarray | None:
        return self._map

    def update(self, diff: np.ndarray, H: np.ndarray | None) -> np.ndarray:
        d = diff.astype(np.float32)
        if self._map is None:
            self._map = d.copy()
            return self._map
        prev = self._map
        if H is not None and prev.shape == diff.shape:
            prev = warp_like(prev, H, diff)
        a = float(self.cfg.persistence_alpha)
        self._map = a * prev + (1.0 - a) * d
        if self.cfg.persistence_decay_floor > 0:
            self._map = np.clip(self._map - self.cfg.persistence_decay_floor,
                                0.0, None)
        return self._map
