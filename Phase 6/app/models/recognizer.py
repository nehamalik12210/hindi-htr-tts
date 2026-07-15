"""
Phase 6 — TrOCR word recognizer (Fixes 6, 9 applied).
Uses beam search with validated parameters. Correct vocab decoding.
Preprocessing ported from trocr base finetune 3.ipynb Cell 5.
"""

import torch
import numpy as np
from PIL import Image
from app.config import (
    TROCR_MAX_LENGTH, TROCR_NUM_BEAMS,
    TROCR_NO_REPEAT_NGRAM, TROCR_BATCH_SIZE,
)
from app.pipeline.preprocessor import preprocess_crop_for_trocr
import unicodedata


def decode_ids(ids, token_to_char, bos_id, pad_id, eos_id):
    """
    Decode token IDs to Hindi string.
    Ported EXACTLY from trocr base finetune 3.ipynb Cell 2.
    """
    out = []
    for tok in ids:
        tok = int(tok)
        if tok == eos_id:
            break
        if tok in (pad_id, bos_id):
            continue
        ch = token_to_char.get(tok, "")
        if ch:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


@torch.inference_mode()
def recognize_words(
    crops: list[Image.Image],
    models: dict,
) -> list[str]:
    """
    Recognize a batch of word crop images using TrOCR.
    Returns list of decoded Hindi strings.
    Ported from trocr base finetune 3.ipynb Cell 7 recognize_batch().
    """
    trocr = models["trocr"]
    processor = models["processor"]
    token_to_char = models["token_to_char"]
    bos_id = models["bos_id"]
    pad_id = models["pad_id"]
    eos_id = models["eos_id"]
    device = models["device"]

    if not crops:
        return []

    # Preprocess all crops using the validated pipeline
    pv_list = []
    for crop in crops:
        pv = preprocess_crop_for_trocr(crop, processor)
        pv_list.append(pv)

    all_texts = []

    # Process in batches (notebook uses batch_size=16)
    batch_size = min(TROCR_BATCH_SIZE, 16)
    for i in range(0, len(pv_list), batch_size):
        batch = torch.stack(pv_list[i:i + batch_size]).to(device)

        # Fix 9 — explicit beam search with validated parameters
        ids = trocr.generate(
            pixel_values=batch,
            max_length=TROCR_MAX_LENGTH,
            num_beams=TROCR_NUM_BEAMS,
            no_repeat_ngram_size=TROCR_NO_REPEAT_NGRAM,
            decoder_start_token_id=bos_id,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

        # Decode each sequence
        for seq in ids:
            text = decode_ids(seq.cpu().tolist(), token_to_char, bos_id, pad_id, eos_id)
            all_texts.append(text)

    return all_texts
