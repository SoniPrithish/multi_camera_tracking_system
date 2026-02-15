"""
YOLOv8 person detector using Ultralytics — optimized for high throughput.

Supports:
  - FP16 half-precision inference on GPU
  - Batch inference (multiple frames in one call)
  - Person-only filtering (class 0)
  - Tunable conf/NMS

Returns detections in the pipeline's standard format:
    [{'bbox': (x, y, w, h), 'score': float, 'cls': int}, ...]
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)


class YOLOv8Detector:
    """Real YOLOv8 detector via ultralytics library — GPU optimized."""

    def __init__(
        self,
        model: str = "yolov8n.pt",
        device: str = "0",
        conf: float = 0.35,
        iou: float = 0.45,
        img_size: int = 640,
        person_only: bool = True,
        half: bool = True,
    ):
        from ultralytics import YOLO

        self.device = device
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.person_only = person_only
        self.half = half and device != "cpu"

        logger.info(
            "Loading YOLOv8 model=%s device=%s conf=%.2f iou=%.2f img=%d half=%s",
            model, device, conf, iou, img_size, self.half,
        )
        self.model = YOLO(model)

        # Warm-up with FP16
        _dummy = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        self.model.predict(
            _dummy, device=device, verbose=False, imgsz=img_size, half=self.half,
        )
        logger.info("YOLOv8 model loaded and warmed up (FP16=%s).", self.half)

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run inference on a single BGR frame."""
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            classes=[0] if self.person_only else None,
            verbose=False,
            half=self.half,
        )
        return self._parse_results(results)

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Run batch inference on multiple frames at once.

        Returns list of detection lists (one per frame).
        """
        if not frames:
            return []

        results = self.model.predict(
            frames,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            classes=[0] if self.person_only else None,
            verbose=False,
            half=self.half,
        )

        batch_dets = []
        for r in results:
            batch_dets.append(self._parse_single(r))
        return batch_dets

    def _parse_results(self, results) -> List[Dict[str, Any]]:
        dets = []
        for r in results:
            dets.extend(self._parse_single(r))
        return dets

    def _parse_single(self, r) -> List[Dict[str, Any]]:
        dets = []
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return dets
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if self.person_only and cls_id != 0:
                continue
            score = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            x, y, w, h = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
            dets.append({"bbox": (x, y, w, h), "score": score, "cls": cls_id})
        return dets
