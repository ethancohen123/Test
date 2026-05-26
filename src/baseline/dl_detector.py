"""Deep-learning detector block.

Wraps a pretrained YOLO model (any Ultralytics `.pt`) behind the same
``list[Detection]`` interface the rest of the pipeline already expects.
Swapping detectors is a one-line config change — no other module knows
or cares whether the source is a top-hat blob, a motion-persistence
blob, or a neural network.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .detector import Detection


@dataclass
class DLDetectorConfig:
    weights: str = "weights/yolov8_thermal.pt"
    conf: float = 0.05           # detector confidence floor
    imgsz: int = 640             # inference size
    classes: tuple[int, ...] | None = None   # None = all classes
    iou: float = 0.45            # NMS IoU
    max_det: int = 30
    top_k: int = 10              # cap returned candidates
    min_box_side: int = 4        # discard sub-pixel boxes


class DLDetector:
    """Thin adapter over `ultralytics.YOLO` — no learning, just inference."""

    def __init__(self, cfg: DLDetectorConfig | None = None):
        self.cfg = cfg or DLDetectorConfig()
        weights_path = Path(self.cfg.weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Detector weights not found: {weights_path.resolve()}. "
                "Place a YOLOv8 .pt file there, e.g. weights/yolov8_thermal.pt"
            )
        # Lazy-import so the rest of the project does not require torch /
        # ultralytics for the classical baselines.
        from ultralytics import YOLO
        self.model = YOLO(str(weights_path))
        self.names: dict[int, str] = dict(self.model.names)

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        kwargs = dict(conf=self.cfg.conf, imgsz=self.cfg.imgsz,
                       iou=self.cfg.iou, max_det=self.cfg.max_det,
                       verbose=False)
        if self.cfg.classes is not None:
            kwargs["classes"] = list(self.cfg.classes)
        results = self.model(frame_bgr, **kwargs)
        if not results:
            return []
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []
        dets: list[Detection] = []
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            w, h = x2 - x1, y2 - y1
            if w < self.cfg.min_box_side or h < self.cfg.min_box_side:
                continue
            score = float(box.conf[0])
            dets.append(Detection(bbox=(int(round(x1)), int(round(y1)),
                                          int(round(w)), int(round(h))),
                                    score=score, area=int(round(w * h))))
        dets.sort(key=lambda d: d.score, reverse=True)
        return dets[: self.cfg.top_k]
