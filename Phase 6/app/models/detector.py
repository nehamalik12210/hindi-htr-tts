"""
Phase 6 — YOLO word detector (Fix 10 — validated padding, no invented IoU-merging).
Line grouping ported from trocr base finetune 3.ipynb Cell 7.
"""

import cv2
import numpy as np
from app.config import (
    YOLO_IMGSZ, YOLO_CONF, YOLO_IOU,
    YOLO_PAD_X, YOLO_PAD_Y_FRAC, YOLO_PAD_Y_MIN, YOLO_PAD_Y_MAX,
)


def detect_words(img_array: np.ndarray, yolo_model) -> list[dict]:
    """
    Run YOLO detection on a page image (expects RGB numpy array).
    Returns list of boxes: [{"x1": int, "y1": int, "x2": int, "y2": int}, ...]
    Ported from trocr base finetune 3.ipynb Cell 7 detect_boxes().
    """
    h, w = img_array.shape[:2]

    # YOLO expects BGR
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array

    det_results = yolo_model(
        img_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False
    )

    boxes = []
    if det_results[0].boxes is not None:
        for box in det_results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box[:4]
            bh = y2 - y1

            # Validated asymmetric padding
            pad_x = YOLO_PAD_X
            pad_y = max(YOLO_PAD_Y_MIN, min(YOLO_PAD_Y_MAX, int(YOLO_PAD_Y_FRAC * bh)))

            x1 = max(0, int(x1) - pad_x)
            y1 = max(0, int(y1) - pad_y)
            x2 = min(w, int(x2) + pad_x)
            y2 = min(h, int(y2) + pad_y)

            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    return boxes


def group_into_lines(boxes: list[dict], words: list[str]) -> list[str]:
    """
    Group detected word boxes into text lines using median-height clustering.
    Ported EXACTLY from trocr base finetune 3.ipynb Cell 7 group_into_lines().
    Returns list of line strings.
    """
    if not boxes:
        return []

    heights = [b["y2"] - b["y1"] for b in boxes]
    med_h = max(10, float(np.median(heights)))
    thr = med_h * 0.6

    items = list(zip(boxes, words))
    items.sort(key=lambda bw: (bw[0]["y1"] + bw[0]["y2"]) / 2.0)

    clusters = []
    current = [items[0]]
    current_yc = (items[0][0]["y1"] + items[0][0]["y2"]) / 2.0

    for b, w in items[1:]:
        yc = (b["y1"] + b["y2"]) / 2.0
        if abs(yc - current_yc) < thr:
            current.append((b, w))
            # Running average of y-centers
            current_yc = float(np.mean([(bb["y1"] + bb["y2"]) / 2.0 for bb, _ in current]))
        else:
            clusters.append(current)
            current = [(b, w)]
            current_yc = yc

    clusters.append(current)

    # Sort clusters by vertical position
    clusters.sort(key=lambda cl: float(np.mean([(bb["y1"] + bb["y2"]) / 2.0 for bb, _ in cl])))

    # Within each cluster, sort left-to-right and join
    lines = []
    for cl in clusters:
        cl_sorted = sorted(cl, key=lambda bw: bw[0]["x1"])
        line_text = " ".join(w for _, w in cl_sorted if w)
        if line_text:
            lines.append(line_text)

    return lines
