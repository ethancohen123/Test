"""Run a baseline pipeline on the full clip and write an annotated mp4.

Default pipeline is the motion-based v2 (fully unsupervised, ego-motion
compensated). Pass --pipeline intensity for the v1 top-hat baseline.

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
from baseline.pipeline import (DLPipeline, DLPipelineConfig,  # noqa: E402
                                FollowingPipeline, FollowingPipelineConfig,
                                HybridPipeline, HybridPipelineConfig,
                                IDPipeline, IDPipelineConfig,
                                MotionPipeline, MotionPipelineConfig,
                                Pipeline, PipelineConfig)
from baseline.tracker import TrackState  # noqa: E402
from baseline.visualize import (draw_candidates, draw_hud, draw_identity_legend,  # noqa: E402
                                  draw_identity_tracks, draw_legend,
                                  draw_track, persistence_heatmap, side_by_side)


def _build_pipeline(args):
    """Return (pipeline, side_panel_kind) where side_panel_kind is one of
    'persistence' | 'gray' | 'detections'."""
    if args.pipeline == "motion":
        return MotionPipeline(MotionPipelineConfig()), "persistence"
    if args.pipeline == "dl":
        from baseline.dl_detector import DLDetectorConfig
        det_cfg = DLDetectorConfig(weights=args.dl_weights,
                                    conf=args.dl_conf,
                                    imgsz=args.dl_imgsz)
        return DLPipeline(DLPipelineConfig(dl_detector=det_cfg)), "gray"
    if args.pipeline == "hybrid":
        from baseline.dl_detector import DLDetectorConfig
        det_cfg = DLDetectorConfig(weights=args.dl_weights,
                                    conf=args.dl_conf,
                                    imgsz=args.dl_imgsz)
        return (HybridPipeline(HybridPipelineConfig(dl_detector=det_cfg)),
                "persistence")
    if args.pipeline == "follow":
        from baseline.dl_detector import DLDetectorConfig
        det_cfg = DLDetectorConfig(weights=args.dl_weights,
                                    conf=args.dl_conf,
                                    imgsz=args.dl_imgsz)
        return (FollowingPipeline(FollowingPipelineConfig(dl_detector=det_cfg)),
                "persistence")
    if args.pipeline == "ids":
        from baseline.dl_detector import DLDetectorConfig
        det_cfg = DLDetectorConfig(weights=args.dl_weights,
                                    conf=args.dl_conf,
                                    imgsz=args.dl_imgsz)
        return (IDPipeline(IDPipelineConfig(dl_detector=det_cfg)), "gray")
    cfg = PipelineConfig(init_frame_index=args.init_frame)
    if args.init_bbox:
        cfg.init_bbox = tuple(int(v) for v in args.init_bbox.split(","))  # type: ignore[assignment]
    return Pipeline(cfg), "gray"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", default="outputs/baseline.mp4")
    p.add_argument("--pipeline",
                    choices=["motion", "intensity", "dl", "hybrid",
                              "follow", "ids"],
                    default="motion")
    p.add_argument("--dl-weights", default="weights/yolov8_thermal.pt",
                    help="(dl pipeline) path to .pt weights")
    p.add_argument("--dl-conf", type=float, default=0.10,
                    help="(dl pipeline) detector confidence floor")
    p.add_argument("--dl-imgsz", type=int, default=640,
                    help="(dl pipeline) inference image size")
    p.add_argument("--side-by-side", action="store_true",
                    help="Pipe-aware side panel: persistence heatmap for "
                          "motion, CLAHE gray for intensity")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--init-frame", type=int, default=0,
                    help="(intensity pipeline) frame at which to seed track")
    p.add_argument("--init-bbox", type=str, default=None,
                    help="(intensity pipeline) manual init bbox 'x,y,w,h'")
    args = p.parse_args()

    meta = probe(args.video)
    print(f"video: {args.video} {meta.width}x{meta.height} @ {meta.fps:.1f}fps "
          f"({meta.n_frames} frames)  pipeline={args.pipeline}")

    pipe, side_kind = _build_pipeline(args)
    trail: list[tuple[int, int]] = []
    writer = VideoWriter(args.out, fps=meta.fps)

    counts = {s: 0 for s in TrackState}
    n_switches = 0
    n = 0
    t0 = time.time()
    stop = args.max_frames or None
    id_max_seen = 0   # multi-track only
    with writer:
        for idx, frame in iter_frames(args.video, stop=stop):
            res = pipe.step(idx, frame)
            if res.modality_switched:
                n_switches += 1
                trail.clear()

            # Two render paths: multi-ID pipeline vs single-target pipelines.
            if args.pipeline == "ids":
                annotated = draw_candidates(frame, res.candidates)
                annotated = draw_identity_tracks(annotated, res.tracks)
                # Count active tracks by state for the stats line.
                for t in res.tracks:
                    counts[t.state] = counts.get(t.state, 0) + 1
                    id_max_seen = max(id_max_seen, t.id)
                annotated = draw_hud(annotated, idx,
                                      len(res.candidates), None)
                if args.side_by_side:
                    annotated = side_by_side(res.gray, annotated)
                lost_count = len(getattr(pipe.id_tracker, 'lost', []))
                annotated = draw_identity_legend(annotated, res.tracks,
                                                   lost_count=lost_count)
            else:
                annotated = draw_candidates(frame, res.candidates)
                if res.track is not None:
                    x, y, w, h = res.track.bbox
                    trail.append((x + w // 2, y + h // 2))
                    if len(trail) > 60:
                        trail = trail[-60:]
                    annotated = draw_track(annotated, res.track, trail)
                    counts[res.track.state] = counts.get(res.track.state, 0) + 1
                annotated = draw_hud(annotated, idx, len(res.candidates),
                                      res.track)
                if args.side_by_side:
                    if side_kind == "persistence":
                        left = persistence_heatmap(res.persistence,
                                                    shape=frame.shape[:2])
                    else:
                        left = res.gray
                    annotated = side_by_side(left, annotated)
                annotated = draw_legend(annotated)
            writer.write(annotated)
            n += 1

    dt = time.time() - t0
    print(f"processed {n} frames in {dt:.2f}s "
          f"({n / max(dt, 1e-3):.1f} fps)  modality switches: {n_switches}")
    total_track = sum(counts.values()) or 1
    for s, c in counts.items():
        print(f"  {s.name:9s}  {c:4d}  ({100 * c / total_track:5.1f}%)")
    if args.pipeline == "ids":
        print(f"  IDs assigned: {id_max_seen}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
