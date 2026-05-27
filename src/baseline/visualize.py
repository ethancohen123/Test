"""Overlay helpers for the annotated output frames.

Colour scheme (BGR):
    DETECTOR candidates  → cyan, thin (1 px), no label
    TRACK state TRACKING → bright green, thick (3 px)
    TRACK state COASTING → orange, thick (3 px), dashed
    TRACK state LOST     → magenta, thick (3 px), dashed
A legend block is drawn in the top-left corner so the viewer never has
to guess which box is which.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detector import Detection
from .tracker import TrackState, TrackState_


# BGR
CAND_COLOR = (255, 255, 0)        # cyan
TRACK_COLOR = {
    TrackState.TRACKING: (0, 230, 0),     # bright green
    TrackState.COASTING: (0, 165, 255),   # orange
    TrackState.LOST:     (220, 0, 220),   # magenta
    TrackState.INIT:     (180, 180, 180), # grey
}
HUD_TEXT = (240, 240, 240)


def _dashed_rect(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
                  color, thickness: int = 2, dash: int = 6) -> None:
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash * 2):
        cv2.line(img, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_candidates(frame: np.ndarray, dets: list[Detection],
                    color=CAND_COLOR, thickness: int = 1) -> np.ndarray:
    out = frame.copy()
    for d in dets:
        x, y, w, h = d.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def draw_track(frame: np.ndarray, ts: TrackState_,
               trail: list[tuple[int, int]] | None = None) -> np.ndarray:
    out = frame.copy()
    color = TRACK_COLOR.get(ts.state, (255, 255, 255))
    x, y, w, h = ts.bbox
    if ts.state in (TrackState.COASTING, TrackState.LOST):
        _dashed_rect(out, (x, y), (x + w, y + h), color, thickness=3)
    else:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
    # Label background for readability.
    label = f"TRACK: {ts.state.name}  s={ts.score:.2f}"
    if ts.state == TrackState.COASTING:
        label += f"  coast={ts.coast_frames}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ly = max(y - 8, th + 4)
    cv2.rectangle(out, (x, ly - th - 4), (x + tw + 6, ly + 4), (0, 0, 0), -1)
    cv2.putText(out, label, (x + 3, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if trail and len(trail) > 1:
        for i in range(1, len(trail)):
            cv2.line(out, trail[i - 1], trail[i], color, 1, cv2.LINE_AA)
    return out


def draw_legend(frame: np.ndarray) -> np.ndarray:
    """Persistent legend block so viewers can decode the colours."""
    out = frame.copy()
    rows = [
        ("DETECTOR candidate", CAND_COLOR, False),
        ("TRACK: TRACKING", TRACK_COLOR[TrackState.TRACKING], False),
        ("TRACK: COASTING", TRACK_COLOR[TrackState.COASTING], True),
        ("TRACK: LOST",     TRACK_COLOR[TrackState.LOST],     True),
    ]
    pad = 6
    line_h = 18
    box_w = 16
    label_w = 170
    panel_w = pad + box_w + 6 + label_w + pad
    panel_h = pad + line_h * len(rows) + pad
    # semi-transparent black background
    overlay = out.copy()
    cv2.rectangle(overlay, (4, 4), (4 + panel_w, 4 + panel_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
    for i, (text, color, dashed) in enumerate(rows):
        y = 4 + pad + i * line_h + line_h - 4
        bx1 = 4 + pad
        bx2 = bx1 + box_w
        by1 = y - line_h + 6
        by2 = y - 2
        if dashed:
            _dashed_rect(out, (bx1, by1), (bx2, by2), color, thickness=2, dash=3)
        else:
            cv2.rectangle(out, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(out, text, (bx2 + 6, y - 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_TEXT, 1, cv2.LINE_AA)
    return out


def draw_hud(frame: np.ndarray, frame_idx: int, n_dets: int,
             ts: TrackState_ | None) -> np.ndarray:
    out = frame.copy()
    h = out.shape[0]
    lines = [f"frame {frame_idx}", f"dets  {n_dets}"]
    if ts is not None:
        lines.append(f"state {ts.state.name}")
    y = h - 8 - 14 * (len(lines) - 1)
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    HUD_TEXT, 1, cv2.LINE_AA)
        y += 14
    return out


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.ndim == 2:
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    if right.ndim == 2:
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    if left.shape[0] != right.shape[0]:
        scale = right.shape[0] / left.shape[0]
        left = cv2.resize(left, (int(left.shape[1] * scale), right.shape[0]))
    return np.hstack([left, right])


def persistence_heatmap(pmap: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if pmap is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    m = pmap.astype(np.float32)
    m = np.clip(m, 0, None)
    mx = float(m.max()) if m.size else 0.0
    if mx > 1e-3:
        m = (255.0 * m / mx).astype(np.uint8)
    else:
        m = np.zeros_like(m, dtype=np.uint8)
    return cv2.applyColorMap(m, cv2.COLORMAP_JET)


# --------------------------------------------------------------------------- #
# ID-aware drawing (multi-target identity pipeline)
# --------------------------------------------------------------------------- #


_PHI_INV = 0.6180339887498949   # golden-ratio conjugate, spreads hues nicely


def id_to_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic vivid BGR colour for a given integer ID.

    Uses a golden-ratio hop on hue so consecutive IDs land far apart
    on the colour wheel. Saturation and value are fixed-bright so the
    colours stand out on dim thermal backgrounds.
    """
    import colorsys
    h = (track_id * _PHI_INV) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_identity_tracks(frame: np.ndarray, tracks) -> np.ndarray:
    """Draw each track with its unique ID colour and label.

    Solid box for TRACKING; dashed for COASTING; LOST tracks aren't in
    the active list so they don't appear here.
    """
    out = frame.copy()
    for t in tracks:
        col = id_to_color(t.id)
        x, y, w, h = t.bbox
        if t.state == TrackState.COASTING:
            _dashed_rect(out, (x, y), (x + w, y + h), col, thickness=2)
        else:
            cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        label = f"ID {t.id}"
        if t.state == TrackState.COASTING:
            label += f"  coast={t.coast_frames}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(y - 6, th + 4)
        cv2.rectangle(out, (x, ly - th - 4), (x + tw + 6, ly + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (x + 3, ly),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return out


def draw_identity_legend(frame: np.ndarray, tracks,
                           lost_count: int = 0) -> np.ndarray:
    """Compact legend for the ID pipeline: shows count of active
    tracks, lost tracks, and a tiny colour swatch per active ID."""
    out = frame.copy()
    pad = 6
    line_h = 16
    n_lines = 2 + min(len(tracks), 6)
    panel_w = 150
    panel_h = pad + line_h * n_lines + pad
    overlay = out.copy()
    cv2.rectangle(overlay, (4, 4), (4 + panel_w, 4 + panel_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)
    y = 4 + pad + line_h - 4
    cv2.putText(out, f"active {len(tracks)}  lost {lost_count}",
                 (4 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                 HUD_TEXT, 1, cv2.LINE_AA)
    y += line_h
    cv2.putText(out, "IDs:", (4 + pad, y),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_TEXT, 1, cv2.LINE_AA)
    y += line_h
    for t in tracks[:6]:
        col = id_to_color(t.id)
        cv2.rectangle(out, (4 + pad, y - 10), (4 + pad + 14, y - 2), col, -1)
        cv2.putText(out, f"#{t.id}", (4 + pad + 20, y - 2),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, HUD_TEXT, 1, cv2.LINE_AA)
        y += line_h
    return out
