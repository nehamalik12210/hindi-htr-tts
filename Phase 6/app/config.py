"""
Phase 6 — Configuration constants.
All validated values ported directly from tts_integrated.ipynb and
yolo detection and label generation.ipynb.
"""

# ── HuggingFace Model Repository IDs ──────────────────────────
HF_YOLO_REPO = "Neha12210/hindi-htr-yolo"
HF_YOLO_FILE = "yolo_best.pt"

HF_TROCR_REPO = "Neha12210/hindi-htr-trocr"
HF_TROCR_FILE = "trocr_base_finetune_3.pt"
HF_ARTIFACTS_FILE = "session2_artifacts.json"

HF_TROCR_BASE = "microsoft/trocr-base-handwritten"
HF_PARLER_MODEL = "ai4bharat/indic-parler-tts"

# ── TrOCR Generation Parameters (validated in Phase 4) ────────
TROCR_IMG_SIZE = 384
TROCR_MAX_LENGTH = 48
TROCR_NUM_BEAMS = 4
TROCR_NO_REPEAT_NGRAM = 0
TROCR_BATCH_SIZE = 32

# ── YOLO Detection Parameters (validated in Phase 4) ──────────
YOLO_IMGSZ = 1024
YOLO_CONF = 0.3
YOLO_IOU = 0.5
YOLO_PAD_X = 6
YOLO_PAD_Y_FRAC = 0.12
YOLO_PAD_Y_MIN = 1
YOLO_PAD_Y_MAX = 3

# ── Parler-TTS Voice Descriptions (Fix 5 — named speakers) ────
PARLER_FEMALE_DESC = (
    "Divya's voice is clear and warm, speaking Hindi at a moderate pace. "
    "The recording is of very high quality, with no background noise."
)
PARLER_MALE_DESC = (
    "Rohit's voice is clear and steady, speaking Hindi at a moderate pace. "
    "The recording is of very high quality, with no background noise."
)

# ── TTS Pipeline Constants ─────────────────────────────────────
TTS_SAMPLE_RATE = 44100
TTS_MAX_CHUNK_WORDS = 25

# ── Orientation Detection ──────────────────────────────────────
ORIENTATION_ANGLES = [0, 90, 180, 270]
ORIENTATION_MIN_REL_MARGIN = 0.10

# ── File Upload Limits ─────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
ALLOWED_DOC_EXTS = {".pdf", ".docx"}
ALLOWED_EXTS = ALLOWED_IMAGE_EXTS | ALLOWED_DOC_EXTS
