"""
Phase 6 — Page assembler (Fix 7 — orientation correction).
Orchestrates: orient → preprocess_page → detect → crop → recognize → group lines.
Ported EXACTLY from trocr base finetune 3.ipynb Cells 5-7.
"""

import numpy as np
import cv2
import unicodedata
from PIL import Image
from app.config import ORIENTATION_MIN_REL_MARGIN
from app.pipeline.preprocessor import preprocess_page
from app.models.detector import detect_words, group_into_lines
from app.models.recognizer import recognize_words


def get_orientation_scores(page_bgr, yolo_model):
    """
    Score all 4 rotations using YOLO confidence × aspect ratio.
    Ported EXACTLY from trocr base finetune 3.ipynb Cell 6.
    """
    rotations = {
        0: page_bgr,
        90: cv2.rotate(page_bgr, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(page_bgr, cv2.ROTATE_180),
        270: cv2.rotate(page_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }
    scores = {}
    for angle, rot in rotations.items():
        det = yolo_model(rot, imgsz=1024, conf=0.3, iou=0.5, verbose=False)
        boxes = det[0].boxes
        if boxes is None or len(boxes) == 0:
            scores[angle] = 0.0
        else:
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            aspect = np.clip(np.mean(w / np.maximum(h, 1)), 0.1, 3.0)
            scores[angle] = float(confs.sum()) * aspect
    return scores


def correct_orientation(img_pil, yolo_model, min_rel_margin=ORIENTATION_MIN_REL_MARGIN):
    """
    Fix 7 — Test all 4 rotations, pick best by YOLO confidence × aspect.
    Ported EXACTLY from trocr base finetune 3.ipynb Cell 6.
    """
    arr_bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    scores = get_orientation_scores(arr_bgr, yolo_model)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_angle, best_score = ranked[0]
    _, second_score = ranked[1]
    rel_margin = (best_score - second_score) / (best_score + 1e-6)

    if best_angle == 0 or rel_margin < min_rel_margin:
        return img_pil

    # Use PIL transpose (matches notebook exactly)
    rot_map = {
        90: Image.ROTATE_270,
        180: Image.ROTATE_180,
        270: Image.ROTATE_90,
    }
    return img_pil.transpose(rot_map[best_angle])


def process_page(img_pil: Image.Image, models: dict) -> dict:
    """
    Full page processing pipeline — ported from trocr base finetune 3.ipynb Cell 7 page_pipeline().

    Steps (exactly matching notebook):
    1. correct_orientation(img)
    2. arr = preprocess_page(img)  ← grayscale → autocontrast → deskew → RGB
    3. boxes = detect_boxes(arr)
    4. crops from arr (the preprocessed array, NOT original image)
    5. crops_pil = [Image.fromarray(c) for c in crops]
    6. words = recognize_words(crops_pil, models)
    7. lines = group_into_lines(boxes, words)
    8. text = "\\n".join(lines)
    """
    yolo = models["yolo"]

    # Step 1: Orientation correction
    img_pil = correct_orientation(img_pil, yolo)

    # Step 2: Page preprocessing (grayscale → autocontrast → deskew → RGB)
    arr = preprocess_page(img_pil)

    # Step 3: Detect word boxes on preprocessed array
    boxes = detect_words(arr, yolo)

    if not boxes:
        return {"text": "", "word_count": 0, "boxes_detected": 0}

    # Step 4-5: Crop words from preprocessed array → PIL
    crops_pil = []
    for b in boxes:
        crop = arr[b["y1"]:b["y2"], b["x1"]:b["x2"]]
        crops_pil.append(Image.fromarray(crop))

    # Step 6: Recognize all crops
    words = recognize_words(crops_pil, models)

    # Step 7: Group into lines (notebook's algorithm with median-height threshold)
    lines = group_into_lines(boxes, words)

    # Step 8: Join lines
    full_text = "\n".join(lines)

    # NFC normalize
    full_text = unicodedata.normalize("NFC", full_text)

    return {
        "text": full_text,
        "word_count": len(full_text.split()),
        "boxes_detected": len(boxes),
    }
