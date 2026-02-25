"""DeepLabV3 semantic segmentation with colored overlay and class labels."""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

try:
    import torch
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

COCO_LABELS = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor"
]

# Distinct colors for segmentation overlay (BGR)
SEG_COLORS = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128),
    (128, 0, 128), (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0),
    (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128),
    (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
]


class DeepLabV3Segmentation:
    """DeepLabV3-based semantic segmentation with colored overlay."""

    def __init__(self, model_name: str = "deeplabv3_resnet50", input_size: Tuple[int, int] = (480, 640)):
        self.input_size = input_size
        self.model = None
        if TORCH_AVAILABLE:
            try:
                self.model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
            except TypeError:
                self.model = deeplabv3_resnet50(pretrained=True)
            self.model.eval()

    def segment(self, frame: np.ndarray) -> Dict:
        """Segment frame; returns segmentation map, class_probs, class_names, colored_overlay."""
        h, w = frame.shape[:2]
        if self.model is None:
            seg = np.zeros((h, w), dtype=np.float32)
            overlay = frame.copy()
            return {"segmentation": seg, "class_probs": None, "class_names": COCO_LABELS, "colored_overlay": overlay}

        img = cv2.resize(frame, (self.input_size[1], self.input_size[0]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        if torch.cuda.is_available():
            x = x.cuda()
            self.model = self.model.cuda()
        with torch.no_grad():
            out = self.model(x)["out"][0]
        pred = out.argmax(0).cpu().numpy().astype(np.uint8)
        probs = torch.softmax(out, 0).cpu().numpy()
        seg_resized = cv2.resize(pred.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        seg_norm = seg_resized.astype(np.float32) / max(len(COCO_LABELS) - 1, 1)

        overlay = _create_colored_overlay(frame, pred, h, w)
        return {
            "segmentation": seg_norm,
            "class_probs": probs,
            "class_names": COCO_LABELS,
            "colored_overlay": overlay,
            "pred_mask": pred,
        }


def _create_colored_overlay(frame: np.ndarray, pred: np.ndarray, h: int, w: int) -> np.ndarray:
    pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = frame.copy().astype(np.float32)
    for c in range(min(len(SEG_COLORS), int(pred.max()) + 1)):
        mask = (pred_full == c).astype(np.float32)
        if mask.sum() == 0:
            continue
        color = np.array(SEG_COLORS[c % len(SEG_COLORS)], dtype=np.float32)
        mask3 = np.stack([mask, mask, mask], axis=-1)
        overlay = overlay * (1 - 0.5 * mask3) + np.broadcast_to(color, overlay.shape) * (0.5 * mask3)
    return np.clip(overlay, 0, 255).astype(np.uint8)
