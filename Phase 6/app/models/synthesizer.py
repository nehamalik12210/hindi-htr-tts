"""
Phase 6 — Indic Parler-TTS synthesizer (Fixes 4, 5 applied).
Uses two tokenizers and named-speaker voice descriptions.
Ported from tts_integrated.ipynb Cells 8/10.
"""

import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from app.config import (
    PARLER_FEMALE_DESC, PARLER_MALE_DESC,
    TTS_SAMPLE_RATE, TTS_MAX_CHUNK_WORDS,
)


def get_voice_description(voice: str) -> str:
    """Return the validated named-speaker description for the given voice."""
    if voice == "male":
        return PARLER_MALE_DESC
    return PARLER_FEMALE_DESC  # default


def synthesize_chunk(
    text: str,
    description: str,
    models: dict,
) -> np.ndarray:
    """
    Synthesize a single text chunk using Parler-TTS.
    Returns numpy audio array.
    """
    parler_model = models["parler_model"]
    parler_tok = models["parler_tokenizer"]
    desc_tok = models["parler_desc_tokenizer"]
    device = models["device"]

    # Fix 4 — description goes through desc_tokenizer, text through parler_tokenizer
    desc_inputs = desc_tok(description, return_tensors="pt").to(device)
    text_inputs = parler_tok(text, return_tensors="pt").to(device)

    with torch.no_grad():
        generation = parler_model.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=text_inputs.input_ids,
            prompt_attention_mask=text_inputs.attention_mask,
        )

    audio_arr = generation.cpu().float().numpy().squeeze()

    # Handle multi-dimensional output
    if audio_arr.ndim > 1:
        audio_arr = audio_arr[0]

    return audio_arr


def chunk_text(text: str) -> list[str]:
    """
    Split text into chunks of max TTS_MAX_CHUNK_WORDS words.
    Splits at sentence boundaries (।, ?, !) first, then by word count.
    Ported from tts_integrated.ipynb Cell 5.
    """
    import re

    # Split at Hindi sentence boundaries
    sentences = re.split(r'(?<=[।?!])\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_count = 0

    for sent in sentences:
        words = sent.split()
        if current_count + len(words) > TTS_MAX_CHUNK_WORDS and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_count = 0

        # If single sentence is too long, split it by word count
        if len(words) > TTS_MAX_CHUNK_WORDS:
            for i in range(0, len(words), TTS_MAX_CHUNK_WORDS):
                sub = words[i : i + TTS_MAX_CHUNK_WORDS]
                chunks.append(" ".join(sub))
        else:
            current_chunk.extend(words)
            current_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]


def synthesize_full(
    text: str,
    voice: str,
    models: dict,
    output_path: str,
) -> dict:
    """
    Full TTS pipeline: normalize → chunk → synthesize → concatenate → save.
    Uses model's native sampling_rate (not hardcoded).
    Returns dict with audio metadata.
    """
    description = get_voice_description(voice)
    chunks = chunk_text(text)

    # Use model's native sample rate (not hardcoded TTS_SAMPLE_RATE)
    sample_rate = models["parler_model"].config.sampling_rate

    audio_segments = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        segment = synthesize_chunk(chunk, description, models)
        audio_segments.append(segment)

        # Add small silence between chunks (0.3s)
        silence = np.zeros(int(sample_rate * 0.3), dtype=np.float32)
        audio_segments.append(silence)

    if not audio_segments:
        audio_segments = [np.zeros(int(sample_rate * 1.0), dtype=np.float32)]

    # Concatenate all segments
    full_audio = np.concatenate(audio_segments)

    # Normalize to prevent clipping (matches notebook per-chunk normalization)
    peak = max(abs(full_audio.max()), abs(full_audio.min()), 1e-6)
    full_audio = full_audio / peak * 0.95

    # Save as WAV using model's native sample rate
    sf.write(output_path, full_audio, sample_rate)

    duration = len(full_audio) / sample_rate
    return {
        "path": output_path,
        "duration_seconds": round(duration, 1),
        "sample_rate": sample_rate,
        "chunks_processed": len(chunks),
        "voice": voice,
    }
