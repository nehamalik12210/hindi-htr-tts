"""
Phase 6 — Preprocessing functions.
Ported EXACTLY from trocr base finetune 3.ipynb Cell 5.
"""

import cv2
import numpy as np
from PIL import Image
from app.config import TROCR_IMG_SIZE


def autocontrast(gray, clip=1.0):
    """Histogram-based autocontrast on grayscale image."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    c = clip / 100 * float(gray.size)
    cs = np.cumsum(hist)
    lo = int(np.searchsorted(cs, c))
    hi = int(255 - np.searchsorted(cs[::-1], c))
    if hi <= lo:
        return gray
    return np.clip((gray.astype(np.float32) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)


def deskew(gray, max_deg=5.0):
    """Hough-line-based deskew on grayscale image."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return gray
    angle = float(np.median([l[0][1] * 180 / np.pi - 90 for l in lines]))
    if abs(angle) > max_deg:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), borderValue=255)


def preprocess_page(img):
    """
    Full page preprocessing: grayscale → autocontrast → deskew → RGB.
    Ported from trocr base finetune 3.ipynb Cell 5.
    """
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    processed = deskew(autocontrast(gray))
    return cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)


def preprocess_crop_for_trocr(crop_pil, processor):
    """
    Word crop preprocessing: grayscale → percentile contrast → Otsu binarize →
    ink bounding box tight crop → aspect-preserve resize → white canvas → processor.
    Ported EXACTLY from trocr base finetune 3.ipynb Cell 5.
    """
    # Convert to grayscale
    gray = np.array(crop_pil.convert("L"))

    # Percentile-based contrast stretch
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 > p2:
        gray = np.clip((gray - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

    # Otsu binarization to find ink region
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(bw)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        y1, y2 = max(0, y - 2), min(gray.shape[0], y + h + 2)
        x1, x2 = max(0, x - 2), min(gray.shape[1], x + w + 2)
        gray = gray[y1:y2, x1:x2]

    # Safety check for tiny/empty crops
    if gray.shape[0] < 4 or gray.shape[1] < 4:
        gray = np.full((32, 128), 255, dtype=np.uint8)

    # Convert back to RGB
    rgb = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

    # Aspect-preserving resize to TROCR_SIZE × TROCR_SIZE
    w, h = rgb.size
    scale = min(TROCR_IMG_SIZE / w, TROCR_IMG_SIZE / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = rgb.resize((nw, nh), Image.LANCZOS)

    # Paste on white canvas
    canvas = Image.new("RGB", (TROCR_IMG_SIZE, TROCR_IMG_SIZE), (255, 255, 255))
    canvas.paste(resized, ((TROCR_IMG_SIZE - nw) // 2, (TROCR_IMG_SIZE - nh) // 2))

    # Run through TrOCR processor
    pixel_values = processor(images=canvas, return_tensors="pt").pixel_values.squeeze(0)
    return pixel_values
