"""Run the v2 motion baseline + DL person-class verifier hybrid.

v2 unchanged. A thermal YOLO is called every K frames as a verifier:
  - if YOLO confirms v2's bbox (IoU >= 0.2) → green check, nothing else
  - if YOLO is silent → grey dash, trust v2 (its motion path was built
    for exactly this case)
  - if YOLO fires elsewhere with no overlap → red cross. After N
    consecutive contradiction ticks, v2 is forcibly reseeded onto the
    top YOLO detection.

Usage:
    python scripts/run_motion_v2_hybrid.py \
        --video assets/How_to_hide_from_a_thermal_drone_Ukraine.mp4 \
        --out outputs/motion_v2_hybrid.mp4 --side-by-side \
        --dl-weights weights/yolov8_thermal.pt --dl-every-k 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baseline_v2.io_utils import VideoWriter, iter_frames, probe  # noqa: E402
from baseline_v2.tracker import TrackState  # noqa: E402
from baseline_v2.visualize import (draw_candidates, draw_hud,  # noqa: E402
                                     draw_track, persistence_heatmap,
                                     side_by_side)
from baseline_v2_hybrid.pipeline import (DLStatus,  # noqa: E402
                                          HybridV2Config, HybridV2Pipeline)
from baseline.dl_detector import DLDetectorConfig  # noqa: E402


# --------------------------------------------------------------------------- #
# Visual additions specific to the hybrid: DL status badge + DL boxes.
# --------------------------------------------------------------------------- #


_STATUS_COLOR = {
    DLStatus.VERIFIED:     (0, 220, 0),       # bright green
    DLStatus.CONTRADICTED: (0, 0, 220),       # red
    DLStatus.SILENT:       (180, 180, 180),   # grey
    DLStatus.SEEDLESS:     (200, 200, 0),     # yellow
    DLStatus.INACTIVE:     (120, 120, 120),   # darker grey
}

_STATUS_TEXT = {
    DLStatus.VERIFIED:     "DL VERIFIED",
    DLStatus.CONTRADICTED: "DL CONTRADICTED",
    DLStatus.SILENT:       "DL silent",
    DLStatus.SEEDLESS:     "DL waiting",
    DLStatus.INACTIVE:     "DL inactive",
}


def draw_dl_badge(frame, status, was_run, reseeded):
    """Top-right corner badge: shows DL verifier state."""
    out = frame.copy()
    h, w = out.shape[:2]
    color = _STATUS_COLOR[status]
    label = _STATUS_TEXT[status]
    if reseeded:
        label = "RESEED → " + label
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    pad = 6
    x2 = w - pad
    x1 = x2 - tw - 2 * pad
    y1 = pad
    y2 = y1 + th + 2 * pad
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 1 if was_run else 0)
    cv2.putText(out, label, (x1 + pad, y2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def draw_dl_candidates(frame, dl_cands):
    """Thin cyan boxes for every DL detection considered this tick."""
    out = frame.copy()
    for d in dl_cands:
        x, y, w, h = d.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 0), 1)
        label = f"DL {d.score:.2f}"
        cv2.putText(out, label, (x, max(y - 4, 10)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                     (255, 255, 0), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs/motion_v2_hybrid.mp4")
    p.add_argument("--side-by-side", action="store_true",
                    help="Show v2's persistence heatmap on the left.")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--dl-weights", default="weights/yolov8_thermal.pt")
    p.add_argument("--dl-conf", type=float, default=0.05)
    p.add_argument("--dl-imgsz", type=int, default=640)
    p.add_argument("--dl-every-k", type=int, default=3,
                    help="Run YOLO once every K frames (K=3 ≈ 10 Hz "
                          "on a 30 fps source)")
    p.add_argument("--contradicted-before-reseed", type=int, default=3,
                    help="Consecutive CONTRADICTED ticks before v2 is "
                          "forcibly reseeded from the top DL detection")
    args = p.parse_args()

    meta = probe(args.video)
    print(f"video: {args.video} {meta.width}x{meta.height} @ {meta.fps:.1f}fps "
          f"({meta.n_frames} frames)  pipeline=motion-v2 + DL verifier "
          f"(K={args.dl_every_k})")

    dl_cfg = DLDetectorConfig(weights=args.dl_weights, conf=args.dl_conf,
                                imgsz=args.dl_imgsz)
    hyb_cfg = HybridV2Config(
        dl=dl_cfg,
        dl_every_k=args.dl_every_k,
        contradicted_before_reseed=args.contradicted_before_reseed,
    )
    pipe = HybridV2Pipeline(hyb_cfg)
    writer = VideoWriter(args.out, fps=meta.fps)
    trail: list[tuple[int, int]] = []

    counts = {s: 0 for s in TrackState}
    status_counts = {s: 0 for s in DLStatus}
    n_reseeds = 0
    n_switches = 0
    n = 0
    t0 = time.time()
    stop = args.max_frames or None

    with writer:
        for idx, frame in iter_frames(args.video, stop=stop):
            res = pipe.step(idx, frame)
            if res.modality_switched:
                n_switches += 1
                trail.clear()
            if res.reseeded:
                n_reseeds += 1
            status_counts[res.dl_status] += 1

            # v2-style annotation: candidates + bbox + HUD
            annotated = draw_candidates(frame, res.candidates)
            if res.track is not None:
                x, y, w, h = res.track.bbox
                trail.append((x + w // 2, y + h // 2))
                if len(trail) > 60:
                    trail = trail[-60:]
                annotated = draw_track(annotated, res.track, trail)
                counts[res.track.state] = counts.get(res.track.state, 0) + 1
            annotated = draw_hud(annotated, idx, len(res.candidates), res.track)

            # Hybrid additions: DL candidate boxes (only on DL ticks) + status
            if res.dl_was_run:
                annotated = draw_dl_candidates(annotated, res.dl_candidates)
            annotated = draw_dl_badge(annotated, res.dl_status,
                                        res.dl_was_run, res.reseeded)

            if args.side_by_side:
                left = persistence_heatmap(res.persistence,
                                            shape=frame.shape[:2])
                annotated = side_by_side(left, annotated)
            writer.write(annotated)
            n += 1

    dt = time.time() - t0
    print(f"processed {n} frames in {dt:.2f}s "
          f"({n / max(dt, 1e-3):.1f} fps)  modality switches: {n_switches}")
    total = sum(counts.values()) or 1
    for s, c in counts.items():
        print(f"  {s.name:9s}  {c:4d}  ({100 * c / total:5.1f}%)")
    print("DL status breakdown:")
    total_status = sum(status_counts.values()) or 1
    for s, c in status_counts.items():
        print(f"  {s.name:13s}  {c:4d}  ({100 * c / total_status:5.1f}%)")
    print(f"  RESEEDS = {n_reseeds}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
