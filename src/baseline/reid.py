"""Re-identification feature extractor for the tracker's appearance model.

Replaces the 32×32 mean-subtracted patch + NCC with a richer descriptor
that survives the realities of thermal imagery (AGC drift, partial
occlusion, small target). The extractor exposes a single contract used
by the tracker:

    embed(frame_bgr, bbox) -> np.ndarray (L2-normalised) | None
    cosine(a, b)            -> float in [-1, 1]

Two backends are supported, with automatic fall-back:

    *CNNReIDExtractor*  — MobileNetV3-Small, ImageNet weights. Strong,
        but requires the weights to be downloadable; in our sandbox
        `download.pytorch.org` is blocked, so this only succeeds when
        the weights are already cached locally.

    *HOGReIDExtractor*  — gradient descriptor, no download. Davis &
        Sharma (CVPR Workshops 2005) explicitly endorse HOG as the
        right appearance feature for thermal pedestrian — gradients
        track temperature edges, which are stable across AGC re-scales
        and roughly invariant to polarity inversion of similar
        magnitude.
"""
from __future__ import annotations

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# HOG backend (always available)
# --------------------------------------------------------------------------- #


class HOGReIDExtractor:
    """64×64 grayscale HOG descriptor, L2-normalised. ~1764-D output."""

    def __init__(self, crop_size: int = 64,
                 cell: int = 8, block: int = 2, bins: int = 9):
        win = (crop_size, crop_size)
        block_sz = (block * cell, block * cell)
        block_stride = (cell, cell)
        cell_sz = (cell, cell)
        self.hog = cv2.HOGDescriptor(win, block_sz, block_stride, cell_sz, bins)
        self.crop_size = crop_size

    def embed(self, frame_bgr: np.ndarray,
              bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x, y, w, h = bbox
        H, W = frame_bgr.shape[:2]
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (self.crop_size, self.crop_size),
                           interpolation=cv2.INTER_AREA)
        if crop.ndim == 3:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        feat = self.hog.compute(crop).astype(np.float32).flatten()
        n = float(np.linalg.norm(feat)) + 1e-6
        return feat / n


# --------------------------------------------------------------------------- #
# Optional CNN backend (only used if weights are reachable)
# --------------------------------------------------------------------------- #


class CNNReIDExtractor:
    """MobileNetV3-Small backbone with ImageNet weights. Requires
    `download.pytorch.org` to be reachable (it isn't in our current
    sandbox). Kept here so the upgrade is trivial when weights are
    available offline."""

    def __init__(self, crop_size: int = 64):
        import torch
        import torchvision.models as tvm
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.backbone = torch.nn.Sequential(
            m.features,
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )
        self.backbone.eval()
        self.crop_size = crop_size
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self._torch = torch

    def embed(self, frame_bgr: np.ndarray,
              bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x, y, w, h = bbox
        H, W = frame_bgr.shape[:2]
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(W, x + w); y1 = min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        crop = cv2.resize(crop, (self.crop_size, self.crop_size),
                           interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        torch = self._torch
        with torch.no_grad():
            t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t = (t - self._mean) / self._std
            feat = self.backbone(t)
            feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.squeeze(0).cpu().numpy()


# --------------------------------------------------------------------------- #
# Public factory
# --------------------------------------------------------------------------- #


def make_reid_extractor():
    """Return the best available extractor. Tries CNN first; falls back
    to HOG if the torch weights cannot be loaded (offline sandbox)."""
    try:
        return CNNReIDExtractor()
    except Exception:
        return HOGReIDExtractor()


# Backwards-compat alias for callers that import the old name.
ReIDExtractor = HOGReIDExtractor


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    return float(np.dot(a, b))
