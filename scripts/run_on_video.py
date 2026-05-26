"""Run the baseline pipeline on the full clip and write an annotated mp4.

Usage:
    python scripts/run_on_video.py --video assets/clip.mp4 \
        --out outputs/baseline.mp4 --side-by-side
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baseline.io_utils import VideoWriter, iter_frames, probe  # noqa: E402
from baseline.pipeline import Pipeline, PipelineConfig  # noqa: E402
from baseline.tracker import TrackState  # noqa: E402
from baseline.visualize import (draw_candidates, draw_hud, draw_track,  # noqa: E402
                                  side_by_side)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs/baseline.mp4")
    p.add_argument("--side-by-side", action="store_true")
    p.add_argument("--max-frames", type=int, default=0,
                    help="Process at most N frames (0 = whole clip)")
    p.add_argument("--init-frame", type=int, default=0,
                    help="Frame index on which to initialise the track")
    p.add_argument("--init-bbox", type=str, default=None,
                    help="Manual init bbox 'x,y,w,h'; overrides auto-detection")
    args = p.parse_args()

    cfg = PipelineConfig(init_frame_index=args.init_frame)
    if args.init_bbox:
        cfg.init_bbox = tuple(int(v) for v in args.init_bbox.split(","))  # type: ignore[assignment]

    meta = probe(args.video)
    print(f"video: {args.video} {meta.width}x{meta.height} @ {meta.fps:.1f}fps "
          f"({meta.n_frames} frames)")

    pipe = Pipeline(cfg)
    trail: list[tuple[int, int]] = []
    writer = VideoWriter(args.out, fps=meta.fps)

    counts = {s: 0 for s in TrackState}
    n = 0
    t0 = time.time()
    stop = args.max_frames or None
    with writer:
        for idx, frame in iter_frames(args.video, stop=stop):
            res = pipe.step(idx, frame)
            annotated = draw_candidates(frame, res.candidates)
            if res.track is not None:
                x, y, w, h = res.track.bbox
                trail.append((x + w // 2, y + h // 2))
                if len(trail) > 60:
                    trail = trail[-60:]
                annotated = draw_track(annotated, res.track, trail)
                counts[res.track.state] = counts.get(res.track.state, 0) + 1
            annotated = draw_hud(annotated, idx, len(res.candidates), res.track)
            if args.side_by_side:
                annotated = side_by_side(res.gray, annotated)
            writer.write(annotated)
            n += 1

    dt = time.time() - t0
    print(f"processed {n} frames in {dt:.2f}s "
          f"({n / max(dt, 1e-3):.1f} fps)")
    total_track = sum(counts.values()) or 1
    for s, c in counts.items():
        print(f"  {s.name:9s}  {c:4d}  ({100 * c / total_track:5.1f}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
