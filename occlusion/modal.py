"""Modal (visible) per-person masks: YOLO11-seg detection + SAM 2 box refinement."""
from dataclasses import dataclass

import cv2
import numpy as np

from .config import WEIGHTS, device

YOLO_WEIGHTS = "yolo11m-seg.pt"
SAM_WEIGHTS = "sam2.1_b.pt"


@dataclass
class Person:
    box: np.ndarray      # xyxy, float
    score: float
    mask: np.ndarray     # HxW bool, visible pixels


class ModalSegmenter:
    def __init__(self, yolo_weights=YOLO_WEIGHTS, sam_weights=SAM_WEIGHTS,
                 conf=0.35, use_sam=True):
        from ultralytics import YOLO, SAM
        # ultralytics downloads to the given path if it doesn't exist
        self.yolo = YOLO(str(WEIGHTS / yolo_weights))
        self.sam = SAM(str(WEIGHTS / sam_weights)) if use_sam else None
        self.conf = conf
        self.dev = device()

    def __call__(self, img_bgr):
        h, w = img_bgr.shape[:2]
        r = self.yolo.predict(img_bgr, classes=[0], conf=self.conf,
                              retina_masks=True, device=self.dev, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        yolo_masks = r.masks.data.cpu().numpy() > 0.5
        masks = yolo_masks
        if self.sam is not None:
            s = self.sam.predict(img_bgr, bboxes=boxes.tolist(),
                                 device=self.dev, verbose=False)[0]
            sam_masks = s.masks.data.cpu().numpy() > 0.5
            # keep SAM mask unless it disagrees wildly with YOLO (e.g. grabbed the background)
            masks = []
            for ym, sm in zip(yolo_masks, sam_masks):
                inter = (ym & sm).sum()
                iou = inter / max((ym | sm).sum(), 1)
                masks.append(sm if iou > 0.5 else ym)
            masks = np.stack(masks)
        masks = _resolve_overlaps(masks, scores)
        return [Person(b, float(s), m) for b, s, m in zip(boxes, scores, masks)
                if m.sum() > 0.0005 * h * w]


def _resolve_overlaps(masks, scores):
    """Each pixel belongs to at most one person: the smaller mask wins
    (a small overlapping mask is usually the person in front)."""
    out = masks.copy()
    areas = masks.reshape(len(masks), -1).sum(1)
    for i in range(len(masks)):
        for j in range(len(masks)):
            if i != j and areas[j] < areas[i]:
                out[i] &= ~masks[j]
    return out


def overlay(img_bgr, persons, alpha=0.5):
    out = img_bgr.copy()
    rng = np.random.default_rng(0)
    for p in persons:
        c = rng.integers(60, 255, 3)
        out[p.mask] = (alpha * c + (1 - alpha) * out[p.mask]).astype(np.uint8)
        cnts, _ = cv2.findContours(p.mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, c.tolist(), 3)
    return out
