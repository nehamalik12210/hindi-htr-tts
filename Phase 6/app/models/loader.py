"""
Phase 6 — Model loader.
RUN-TIME ONLY: loads pre-downloaded weights from local disk into GPU VRAM.
Downloading happens at build-time in modal_app.py's download_models().
"""

import json, torch, os
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    AutoTokenizer,
)
from parler_tts import ParlerTTSForConditionalGeneration

from app.config import (
    HF_YOLO_REPO, HF_YOLO_FILE,
    HF_TROCR_REPO, HF_TROCR_FILE, HF_ARTIFACTS_FILE,
    HF_TROCR_BASE, HF_PARLER_MODEL,
)


def load_all_models():
    """
    Load all three model groups from local HF cache into GPU VRAM.
    Returns a dict with all models, tokenizers, and vocab ready to use.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[loader] Device: {device}")

    # ── 1. YOLO Detector ──────────────────────────────────────
    yolo_path = hf_hub_download(HF_YOLO_REPO, HF_YOLO_FILE)
    yolo_model = YOLO(yolo_path)
    print(f"[loader] YOLO loaded from {yolo_path}")

    # ── 2. TrOCR-Base Recognizer (Fix 6 — correct vocab + config) ─
    trocr_path = hf_hub_download(HF_TROCR_REPO, HF_TROCR_FILE)
    artifacts_path = hf_hub_download(HF_TROCR_REPO, HF_ARTIFACTS_FILE)

    # Load the real artifacts structure (Fix 6)
    with open(artifacts_path, "r", encoding="utf-8") as f:
        art = json.load(f)

    char_to_token = art["char_to_token"]
    token_to_char = {int(k): v for k, v in art["token_to_char"].items()}
    VOCAB_SIZE = art["VOCAB_SIZE"]
    PAD_ID = art["PAD_ID"]
    BOS_ID = art["BOS_ID"]
    EOS_ID = art["EOS_ID"]
    UNK_ID = art["UNK_ID"]

    # Build model from base, resize embeddings, load fine-tuned weights
    trocr_model = VisionEncoderDecoderModel.from_pretrained(HF_TROCR_BASE)
    trocr_model.decoder.resize_token_embeddings(VOCAB_SIZE)

    # Load fine-tuned weights — checkpoint saved as {"model_state_dict": ..., "best_wer": ...}
    ckpt = torch.load(trocr_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = trocr_model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[loader] WARNING — missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[loader] WARNING — unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")

    # Fix 6 — set special token IDs on BOTH configs
    trocr_model.config.pad_token_id = PAD_ID
    trocr_model.config.decoder_start_token_id = BOS_ID
    trocr_model.config.eos_token_id = EOS_ID
    trocr_model.config.bos_token_id = BOS_ID
    trocr_model.decoder.config.pad_token_id = PAD_ID
    trocr_model.decoder.config.bos_token_id = BOS_ID
    trocr_model.decoder.config.eos_token_id = EOS_ID
    trocr_model.decoder.config.decoder_start_token_id = BOS_ID

    trocr_model.to(device).eval()

    # Processor for image preprocessing (TrOCRProcessor replaces deprecated AutoFeatureExtractor)
    processor = TrOCRProcessor.from_pretrained(HF_TROCR_BASE)

    print(f"[loader] TrOCR-Base loaded. VOCAB_SIZE={VOCAB_SIZE}, "
          f"PAD={PAD_ID}, BOS={BOS_ID}, EOS={EOS_ID}")

    # ── 3. Indic Parler-TTS (Fix 4 — two tokenizers) ─────────
    parler_model = ParlerTTSForConditionalGeneration.from_pretrained(
        HF_PARLER_MODEL, torch_dtype=torch.float16
    ).to(device)

    # Fix 4 — spoken text tokenizer
    parler_tokenizer = AutoTokenizer.from_pretrained(HF_PARLER_MODEL)

    # Fix 4 — description tokenizer (for voice control prompts)
    desc_tokenizer_name = parler_model.config.text_encoder._name_or_path
    parler_description_tokenizer = AutoTokenizer.from_pretrained(desc_tokenizer_name)

    print(f"[loader] Parler-TTS loaded. Desc tokenizer: {desc_tokenizer_name}")

    return {
        "device": device,
        # YOLO
        "yolo": yolo_model,
        # TrOCR
        "trocr": trocr_model,
        "processor": processor,
        "char_to_token": char_to_token,
        "token_to_char": token_to_char,
        "vocab_size": VOCAB_SIZE,
        "pad_id": PAD_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "unk_id": UNK_ID,
        # Parler-TTS
        "parler_model": parler_model,
        "parler_tokenizer": parler_tokenizer,
        "parler_desc_tokenizer": parler_description_tokenizer,
    }
