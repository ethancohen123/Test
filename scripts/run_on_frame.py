"""Run the baseline detector on a single frame and save the annotated image.

Usage:
    python scripts/run_on_frame.py --video assets/clip.mp4 --frame 30 \
        --out outputs/frame_30.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baseline.io_utils import read_frame  # noqa: E402
from baseline.pipeline import Pipeline, PipelineConfig  # noqa: E402
from baseline.visualize import draw_candidates, draw_hud, side_by_side  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", default="outputs/frame.jpg")
    p.add_argument("--side-by-side", action="store_true")
    args = p.parse_args()

    frame = read_frame(args.video, args.frame)
    pipe = Pipeline(PipelineConfig())
    res = pipe.step(args.frame, frame)

    annotated = draw_candidates(frame, res.candidates)
    annotated = draw_hud(annotated, res.frame_idx, len(res.candidates),
                          res.track)
    if args.side_by_side:
        annotated = side_by_side(res.gray, annotated)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, annotated)
    print(f"wrote {args.out}  candidates={len(res.candidates)}  "
          f"track={res.track.state.name if res.track else 'NONE'}")


if __name__ == "__main__":
    main()
