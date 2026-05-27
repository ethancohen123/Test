"""Evaluate a pipeline against CVAT XML ground truth.

Computes per-frame IoU, success rate at IoU >= {0.3, 0.5}, mean IoU
over frames where GT exists, precision / recall at IoU >= 0.5, and
mean centre-pixel error. Optionally writes a side-by-side mp4 with
GT (green) and prediction (cyan).

The script is purely additive: it imports the existing pipelines
(baseline / baseline_v2 / baseline_v2_hybrid) via their public
factories and calls ``pipe.step(idx, frame)`` exactly the way the
``run_*`` scripts do. Pipeline code is not modified.

Supported --pipeline values:
    motion_v2          baseline_v2.MotionPipeline       (frozen v2, no DL)
    motion_v2_hybrid   baseline_v2_hybrid.HybridV2Pipe  (v2 + YOLO verifier)
    baseline_motion    baseline.MotionPipeline          (older motion)
    baseline_hybrid    baseline.HybridPipeline          (older hybrid)
    baseline_dl        baseline.DLPipeline              (YOLO only)

Usage:
    python scripts/eval_against_gt.py \
        --video "assets/Enregistrement 2026-05-27 123748è-1shot learning.mp4" \
        --xml   assets/annotations_supporting_video.xml \
        --pipeline motion_v2_hybrid \
        --out   outputs/eval_motion_v2_hybrid.json \
        --vis   outputs/eval_motion_v2_hybrid.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BBox = Tuple[int, int, int, int]  # (x, y, w, h)


# --------------------------------------------------------------------------- #
# CVAT XML parsing
# --------------------------------------------------------------------------- #


def parse_cvat_xml(xml_path: Path, track_id: str = "0") -> Dict[int, BBox]:
    """Return {frame_idx: (x, y, w, h)} for one CVAT track.

    Only keeps boxes whose ``outside`` attribute is 0. The default
    ``track_id="0"`` matches the real annotation in our clip; tracks
    1 and 2 are single-keyframe init mistakes (immediately marked
    outside on the next frame).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: Dict[int, BBox] = {}
    for tr in root.findall("track"):
        if tr.get("id") != track_id:
            continue
        for box in tr.findall("box"):
            if int(box.get("outside", "0")):
                continue
            f = int(box.get("frame"))
            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))
            out[f] = (int(xtl), int(ytl),
                      int(round(xbr - xtl)), int(round(ybr - ytl)))
    return out


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def iou_xywh(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def centre_err(a: BBox, b: BBox) -> float:
    ax = a[0] + a[2] / 2.0
    ay = a[1] + a[3] / 2.0
    bx = b[0] + b[2] / 2.0
    by = b[1] + b[3] / 2.0
    return float(np.hypot(ax - bx, ay - by))


# --------------------------------------------------------------------------- #
# Pipeline factories (mirror the run_* scripts, no duplication of logic)
# --------------------------------------------------------------------------- #


def build_pipeline(name: str, dl_weights: str, dl_conf: float,
                   dl_imgsz: int, dl_every_k: int):
    """Return (pipe, iter_frames_fn, probe_fn).

    Each pipeline lives in its own package; we import lazily so that
    motion-only runs don't pay for torch / ultralytics startup.
    """
    if name == "motion_v2":
        from baseline_v2.io_utils import iter_frames, probe
        from baseline_v2.pipeline import MotionPipeline, MotionPipelineConfig
        return MotionPipeline(MotionPipelineConfig()), iter_frames, probe

    if name == "motion_v2_hybrid":
        from baseline_v2.io_utils import iter_frames, probe
        from baseline_v2_hybrid.pipeline import (HybridV2Config,
                                                  HybridV2Pipeline)
        from baseline.dl_detector import DLDetectorConfig
        dl = DLDetectorConfig(weights=dl_weights, conf=dl_conf,
                              imgsz=dl_imgsz)
        cfg = HybridV2Config(dl=dl, dl_every_k=dl_every_k)
        return HybridV2Pipeline(cfg), iter_frames, probe

    if name == "baseline_motion":
        from baseline.io_utils import iter_frames, probe
        from baseline.pipeline import MotionPipeline, MotionPipelineConfig
        return MotionPipeline(MotionPipelineConfig()), iter_frames, probe

    if name == "baseline_hybrid":
        from baseline.io_utils import iter_frames, probe
        from baseline.pipeline import HybridPipeline, HybridPipelineConfig
        from baseline.dl_detector import DLDetectorConfig
        dl = DLDetectorConfig(weights=dl_weights, conf=dl_conf,
                              imgsz=dl_imgsz)
        return (HybridPipeline(HybridPipelineConfig(dl_detector=dl)),
                iter_frames, probe)

    if name == "baseline_dl":
        from baseline.io_utils import iter_frames, probe
        from baseline.pipeline import DLPipeline, DLPipelineConfig
        from baseline.dl_detector import DLDetectorConfig
        dl = DLDetectorConfig(weights=dl_weights, conf=dl_conf,
                              imgsz=dl_imgsz)
        return (DLPipeline(DLPipelineConfig(dl_detector=dl)),
                iter_frames, probe)

    raise ValueError(f"unknown pipeline: {name}")


# --------------------------------------------------------------------------- #
# Visualisation (GT vs pred overlay)
# --------------------------------------------------------------------------- #


def _draw_eval_overlay(frame, idx, gt: Optional[BBox],
                       pred: Optional[BBox], iou_val: Optional[float]):
    out = frame.copy()
    if gt is not None:
        x, y, w, h = gt
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(out, "GT", (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1,
                    cv2.LINE_AA)
    if pred is not None:
        x, y, w, h = pred
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 200, 0), 2)
        cv2.putText(out, "pred", (x, y + h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1,
                    cv2.LINE_AA)
    txt = f"f={idx}"
    if iou_val is not None:
        txt += f"  IoU={iou_val:.2f}"
    cv2.putText(out, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- #
# Main eval
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--xml", required=True)
    p.add_argument("--pipeline", required=True,
                   choices=["motion_v2", "motion_v2_hybrid",
                            "baseline_motion", "baseline_hybrid",
                            "baseline_dl"])
    p.add_argument("--track-id", default="0",
                   help="CVAT track id to evaluate against (default: 0)")
    p.add_argument("--iou-tp", type=float, default=0.5,
                   help="IoU threshold for TP in precision/recall")
    p.add_argument("--out", default=None,
                   help="Path to write JSON metrics; defaults to "
                        "outputs/eval_<pipeline>.json")
    p.add_argument("--vis", default=None,
                   help="Path to write GT-vs-pred overlay mp4 "
                        "(optional, slower)")
    p.add_argument("--dl-weights", default="weights/yolov8_thermal.pt")
    p.add_argument("--dl-conf", type=float, default=0.05)
    p.add_argument("--dl-imgsz", type=int, default=640)
    p.add_argument("--dl-every-k", type=int, default=3)
    p.add_argument("--mask-top", type=int, default=0,
                   help="Zero out the top N rows of every frame before "
                        "passing it to the pipeline. Use to suppress "
                        "screen-recording chrome (e.g., player UI baked "
                        "into the 1-shot annotated clip).")
    args = p.parse_args()

    out_json = Path(args.out) if args.out else (
        ROOT / "outputs" / f"eval_{args.pipeline}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    gt = parse_cvat_xml(Path(args.xml), track_id=args.track_id)
    if not gt:
        print(f"no GT boxes found for track {args.track_id}", file=sys.stderr)
        sys.exit(1)
    gt_frames = sorted(gt.keys())
    print(f"GT: track {args.track_id}, "
          f"{len(gt)} boxes on frames {gt_frames[0]}..{gt_frames[-1]}")

    pipe, iter_frames, probe = build_pipeline(
        args.pipeline, args.dl_weights, args.dl_conf,
        args.dl_imgsz, args.dl_every_k)
    meta = probe(args.video)
    print(f"video: {meta.width}x{meta.height} @ {meta.fps:.1f}fps "
          f"({meta.n_frames} frames)  pipeline={args.pipeline}")

    writer = None
    if args.vis:
        Path(args.vis).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.vis, fourcc, meta.fps,
                                 (meta.width, meta.height))

    per_frame: list[dict] = []
    n_with_gt = 0
    n_with_pred_and_gt = 0
    sum_iou = 0.0
    sum_centre = 0.0
    n_tp = 0   # IoU >= iou_tp where GT exists
    n_fn = 0   # GT exists, no pred (or IoU < iou_tp)
    n_fp = 0   # pred exists, no GT in that frame
    t0 = time.time()

    for idx, frame in iter_frames(args.video):
        if args.mask_top > 0:
            frame[:args.mask_top, :] = 0
        res = pipe.step(idx, frame)
        pred: Optional[BBox] = None
        if getattr(res, "track", None) is not None:
            pred = tuple(int(v) for v in res.track.bbox)  # type: ignore

        gt_box = gt.get(idx)
        iou_v: Optional[float] = None
        if gt_box is not None:
            n_with_gt += 1
            if pred is not None:
                iou_v = iou_xywh(gt_box, pred)
                sum_iou += iou_v
                sum_centre += centre_err(gt_box, pred)
                n_with_pred_and_gt += 1
                if iou_v >= args.iou_tp:
                    n_tp += 1
                else:
                    n_fn += 1
            else:
                n_fn += 1
        else:
            # No GT in this frame — predictions here neither help nor hurt.
            # We *could* count them as FP, but for a single-target sparse
            # GT the cleaner thing is to ignore them. We log them anyway.
            if pred is not None:
                n_fp += 1

        per_frame.append({"frame": idx,
                          "gt": gt_box, "pred": pred,
                          "iou": iou_v})

        if writer is not None:
            writer.write(_draw_eval_overlay(frame, idx, gt_box, pred, iou_v))

    if writer is not None:
        writer.release()

    dt = time.time() - t0
    fps = (len(per_frame) / max(dt, 1e-3))

    # --- aggregate ---
    mean_iou = sum_iou / max(n_with_pred_and_gt, 1)
    # Mean IoU over GT frames (counts missed frames as IoU=0)
    mean_iou_over_gt = sum_iou / max(n_with_gt, 1)
    succ_03 = sum(1 for r in per_frame
                  if r["gt"] is not None and (r["iou"] or 0.0) >= 0.3)
    succ_05 = sum(1 for r in per_frame
                  if r["gt"] is not None and (r["iou"] or 0.0) >= 0.5)
    success_03 = succ_03 / max(n_with_gt, 1)
    success_05 = succ_05 / max(n_with_gt, 1)
    precision = n_tp / max(n_tp + n_fp, 1)
    recall = n_tp / max(n_tp + n_fn, 1)
    mean_centre = sum_centre / max(n_with_pred_and_gt, 1)

    summary = {
        "pipeline": args.pipeline,
        "video": args.video,
        "xml": args.xml,
        "track_id": args.track_id,
        "n_frames": len(per_frame),
        "n_gt_frames": n_with_gt,
        "n_pred_and_gt": n_with_pred_and_gt,
        "mean_iou_when_predicted": round(mean_iou, 4),
        "mean_iou_over_gt": round(mean_iou_over_gt, 4),
        "success_at_0.3": round(success_03, 4),
        "success_at_0.5": round(success_05, 4),
        "precision_at_0.5": round(precision, 4),
        "recall_at_0.5": round(recall, 4),
        "mean_centre_error_px": round(mean_centre, 2),
        "tp": n_tp, "fp": n_fp, "fn": n_fn,
        "eval_fps": round(fps, 1),
    }

    print("\n=== metrics ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_json.write_text(json.dumps(
        {"summary": summary, "per_frame": per_frame}, indent=2))
    print(f"\nwrote {out_json}")
    if args.vis:
        print(f"wrote {args.vis}")


if __name__ == "__main__":
    main()
