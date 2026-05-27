"""Top-hat + adaptive threshold blob detector with shape/area gating.

Produces ranked candidate detections (bbox + score). Operates on a
preprocessed bright-target grayscale image.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    tophat_ksize: int = 15          # structuring-element side, ~target diameter
    abs_thresh: int = 25            # threshold on top-hat response (0-255)
    min_area: int = 8
    max_area: int = 4000
    min_aspect: float = 0.2         # bbox aspect ratio gate (short/long)
    max_aspect: float = 5.0
    open_ksize: int = 3             # post-threshold morphological opening
    top_k: int = 10                 # keep N strongest candidates


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]   # x, y, w, h
    score: float
    area: int

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.bbox
        return x + w / 2.0, y + h / 2.0


def _local_contrast(gray: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    """LCM-style score: mean intensity inside vs in a surrounding ring."""
    H, W = gray.shape
    pad = max(w, h)
    x0 = max(x - pad, 0); y0 = max(y - pad, 0)
    x1 = min(x + w + pad, W); y1 = min(y + h + pad, H)
    outer = gray[y0:y1, x0:x1].astype(np.float32)
    inner = gray[y:y + h, x:x + w].astype(np.float32)
    if inner.size == 0 or outer.size == 0:
        return 0.0
    # Subtract the inner region's contribution from the outer mean.
    outer_sum = float(outer.sum()) - float(inner.sum())
    outer_n = outer.size - inner.size
    if outer_n <= 0:
        return 0.0
    ring_mean = outer_sum / outer_n
    inner_mean = float(inner.mean())
    return inner_mean - ring_mean


def detect(gray: np.ndarray, cfg: DetectorConfig) -> list[Detection]:
    """Run top-hat + adaptive threshold + CC analysis on a bright-target image."""
    k = max(3, cfg.tophat_ksize | 1)  # force odd, >= 3
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)

    # Combine an absolute floor with an Otsu-on-tophat threshold to be
    # robust to varying scene contrast.
    otsu_thr, _ = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(cfg.abs_thresh, int(otsu_thr))
    _, bw = cv2.threshold(tophat, thr, 255, cv2.THRESH_BINARY)

    if cfg.open_ksize >= 3:
        opk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.open_ksize, cfg.open_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, opk)

    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    dets: list[Detection] = []
    for i in range(1, n_lab):
        x, y, w, h, area = stats[i]
        if area < cfg.min_area or area > cfg.max_area:
            continue
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
        if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
            continue
        score = _local_contrast(gray, x, y, w, h)
        dets.append(Detection(bbox=(int(x), int(y), int(w), int(h)),
                              score=float(score), area=int(area)))

    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[: cfg.top_k]


@dataclass
class MotionDetectorConfig:
    abs_thresh: float = 6.0        # min persistence value (0-255 scale)
    rel_thresh_pct: float = 99.0   # alternative: percentile-based
    min_area: int = 12
    max_area: int = 6000
    min_aspect: float = 0.15
    max_aspect: float = 6.0
    open_ksize: int = 3
    close_ksize: int = 5
    top_k: int = 10


def detect_motion(persistence: np.ndarray,
                   cfg: MotionDetectorConfig) -> list[Detection]:
    """Detect connected blobs on the temporal persistence map.

    Scoring = sum of persistence inside the blob. This naturally
    favours blobs that have been hot for many frames over short flashes.
    """
    if persistence is None:
        return []
    p = persistence
    # Combine an absolute floor with a percentile to adapt to global level.
    pct = float(np.percentile(p, cfg.rel_thresh_pct))
    thr = max(cfg.abs_thresh, pct)
    bw = (p >= thr).astype(np.uint8) * 255
    if cfg.open_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (cfg.open_ksize, cfg.open_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    if cfg.close_ksize >= 3:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (cfg.close_ksize, cfg.close_ksize))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    dets: list[Detection] = []
    for i in range(1, n_lab):
        x, y, w, h, area = stats[i]
        if area < cfg.min_area or area > cfg.max_area:
            continue
        aspect = min(w, h) / max(w, h) if max(w, h) > 0 else 0.0
        if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
            continue
        region = p[y:y + h, x:x + w]
        mean_p = float(region.mean())
        # √area damps the contribution of large warp-residual sheets while
        # still preferring a real blob over a 1-pixel flash.
        score = mean_p * (float(area) ** 0.5)
        dets.append(Detection(bbox=(int(x), int(y), int(w), int(h)),
                              score=score, area=int(area)))
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[: cfg.top_k]
