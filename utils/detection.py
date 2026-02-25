"""YOLO object detection with bounding boxes and labels."""

from __future__ import annotations

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict, Any

DETECTOR_AVAILABLE = False
YOLO = None

try:
    from ultralytics import YOLO as _YOLO
    YOLO = _YOLO
    DETECTOR_AVAILABLE = True
except ImportError:
    pass


def run_yolo_detection(
    image: np.ndarray,
    model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Run YOLO detection with bounding boxes and class labels.

    Returns:
        Tuple of (annotated_image, detections)
        detections: list of {"bbox": (x1,y1,x2,y2), "class": str, "conf": float}
    """
    if not DETECTOR_AVAILABLE or YOLO is None:
        return image.copy(), []

    model = YOLO(model_path)
    results = model(image, conf=conf_threshold, iou=iou_threshold, verbose=False)

    detections = []
    annotated = image.copy()

    if results and len(results) > 0:
        r = results[0]
        boxes = r.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = r.names[cls_id] if hasattr(r, "names") else str(cls_id)
                detections.append({
                    "bbox": tuple(map(float, xyxy)),
                    "class": cls_name,
                    "conf": conf,
                })
        annotated = r.plot()

    return annotated, detections


def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    font_scale: float = 0.5,
    thickness: int = 2,
    box_color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    out = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d["bbox"][:4])
        label = f"{d['class']} {d['conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), box_color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), box_color, -1)
        cv2.putText(out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    return out
