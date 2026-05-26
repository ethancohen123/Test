"""Overlay helpers for the annotated output frames."""
from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection
from .tracker import TrackState, TrackState_


STATE_COLOR = {
    TrackState.TRACKING: (0, 220, 0),     # green
    TrackState.COASTING: (0, 200, 220),   # amber
    TrackState.LOST: (0, 0, 220),         # red
    TrackState.INIT: (200, 200, 200),     # grey
}


def draw_candidates(frame: np.ndarray, dets: list[Detection],
                    color=(120, 120, 255), thickness: int = 1) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        x, y, w, h = d.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def draw_track(frame: np.ndarray, ts: TrackState_,
               trail: list[tuple[int, int]] | None = None) -> np.ndarray:
    out = frame.copy()
    color = STATE_COLOR.get(ts.state, (255, 255, 255))
    x, y, w, h = ts.bbox
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    label = f"{ts.state.name}  s={ts.score:.2f}"
    if ts.state == TrackState.COASTING:
        label += f"  coast={ts.coast_frames}"
    cv2.putText(out, label, (x, max(y - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if trail and len(trail) > 1:
        for i in range(1, len(trail)):
            cv2.line(out, trail[i - 1], trail[i], color, 1, cv2.LINE_AA)
    return out


def draw_hud(frame: np.ndarray, frame_idx: int, n_dets: int,
             ts: TrackState_ | None) -> np.ndarray:
    out = frame.copy()
    h = out.shape[0]
    lines = [f"frame {frame_idx}", f"cands {n_dets}"]
    if ts is not None:
        lines.append(f"track {ts.state.name}")
    y = h - 8 - 14 * (len(lines) - 1)
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (240, 240, 240), 1, cv2.LINE_AA)
        y += 14
    return out


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack a preprocessed gray view next to the annotated BGR frame."""
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    if left.shape[0] != right.shape[0]:
        scale = right.shape[0] / left.shape[0]
        left = cv2.resize(left, (int(left.shape[1] * scale), right.shape[0]))
    return np.hstack([left, right])
