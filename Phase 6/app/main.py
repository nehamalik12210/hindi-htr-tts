"""
Phase 6 — FastAPI application.
Serves the frontend + all API endpoints.
Fix 2 (VRAM management) applied in process and synthesize endpoints.
Fix 9 (platform-agnostic startup) — self-loads models if nothing has
loaded them yet, so this app runs standalone on Lightning AI / a bare
`uvicorn app.main:app`, not just under Modal.
"""

import os, uuid, shutil, torch
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import ALLOWED_EXTS, MAX_UPLOAD_SIZE_MB
from app.pipeline.file_handler import extract_pages
from app.pipeline.page_assembler import process_page
from app.pipeline.text_normalizer import normalize_hindi_text
from app.models.synthesizer import synthesize_full
from app.utils.exporters import export_as_pdf, export_as_docx, export_as_png
from app.utils.audio_converter import convert_audio

# ── App Setup ──────────────────────────────────────────────────
app = FastAPI(title="Hindi HTR-TTS", version="1.0")

# Temp directory for generated files
TEMP_DIR = Path("/tmp/htr_tts_outputs")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# In-memory session store (per-request results)
# In production, use Redis/DB — for a demo, dict is fine
_sessions: dict = {}

# Models ref — set either by modal_app.py's @modal.enter() hook (on Modal),
# or by the startup event below (everywhere else, e.g. Lightning AI).
_models: dict = {}


def set_models(models: dict):
    """Called by modal_app.py after loading models into VRAM (Modal path)."""
    global _models
    _models = models


@app.on_event("startup")
async def _load_models_if_needed():
    """
    Fix 9 — Self-contained model loading.

    On Modal, @modal.enter() already calls set_models() before this app
    is served, so _models is non-empty here and this is a no-op.

    On any other platform (Lightning AI Studio, your own machine, a plain
    `uvicorn app.main:app`), nothing else calls set_models(), so this
    loads the models itself on server startup.
    """
    global _models
    if not _models:
        from app.models.loader import load_all_models
        print("[startup] No models loaded yet — loading now...")
        _models = load_all_models()
        print("[startup] Models loaded. Ready to serve.")


def _get_session_dir(session_id: str) -> Path:
    d = TEMP_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Static Files ──────────────────────────────────────────────
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Root ──────────────────────────────────────────────────────
@app.get("/")
async def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── POST /api/process ─────────────────────────────────────────
@app.post("/api/process")
async def process_document(file: UploadFile = File(...)):
    """Upload a document, run OCR, return recognized text."""
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTS}")

    # Read file
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(400, f"File too large: {size_mb:.1f}MB (max {MAX_UPLOAD_SIZE_MB}MB)")

    # Extract pages
    try:
        pages = extract_pages(contents, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Failed to read file: {str(e)}")

    if not pages:
        raise HTTPException(400, "No pages found in the uploaded file.")

    # Process each page through the OCR pipeline
    session_id = str(uuid.uuid4())[:8]
    session_dir = _get_session_dir(session_id)

    all_results = []
    combined_text = []

    for i, page_img in enumerate(pages):
        result = process_page(page_img, _models)
        result["page_num"] = i + 1
        all_results.append(result)
        if result["text"]:
            combined_text.append(result["text"])

    full_text = "\n".join(combined_text)

    # Fix 2 — clear CUDA cache after OCR, before any TTS
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Store in session
    _sessions[session_id] = {
        "text": full_text,
        "pages": all_results,
        "total_pages": len(pages),
        "is_single_page": len(pages) == 1,
        "audio_paths": {},  # filled by /api/synthesize
    }

    return JSONResponse({
        "session_id": session_id,
        "total_pages": len(pages),
        "pages": all_results,
        "combined_text": full_text,
    })


# ── POST /api/synthesize ──────────────────────────────────────
@app.post("/api/synthesize")
async def synthesize_audio(request: Request):
    """Generate speech from recognized text."""
    body = await request.json()
    session_id = body.get("session_id")
    voice = body.get("voice", "female")

    if not session_id or session_id not in _sessions:
        raise HTTPException(400, "Invalid or expired session_id")

    session = _sessions[session_id]
    text = session["text"]

    if not text.strip():
        raise HTTPException(400, "No text to synthesize")

    # Normalize text for TTS
    normalized = normalize_hindi_text(text)

    # Fix 2 — clear CUDA cache before TTS
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate audio
    session_dir = _get_session_dir(session_id)
    wav_path = str(session_dir / f"audio_{voice}.wav")

    try:
        result = synthesize_full(normalized, voice, _models, wav_path)
    except Exception as e:
        raise HTTPException(500, f"TTS synthesis failed: {str(e)}")

    # Store audio path
    session["audio_paths"][voice] = wav_path

    # Fix 2 — clear after TTS too
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return JSONResponse({
        "session_id": session_id,
        "voice": voice,
        "duration_seconds": result["duration_seconds"],
        "chunks_processed": result["chunks_processed"],
        "audio_url": f"/api/stream/audio/{session_id}/{voice}",
    })


# ── GET /api/stream/audio ─────────────────────────────────────
@app.get("/api/stream/audio/{session_id}/{voice}")
async def stream_audio(session_id: str, voice: str):
    """Stream the generated WAV for the audio player."""
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")

    wav_path = _sessions[session_id]["audio_paths"].get(voice)
    if not wav_path or not Path(wav_path).exists():
        raise HTTPException(404, "Audio not generated yet for this voice")

    return FileResponse(wav_path, media_type="audio/wav")


# ── GET /api/download/text/{format} ───────────────────────────
@app.get("/api/download/text/{fmt}/{session_id}")
async def download_text(fmt: str, session_id: str):
    """Download recognized text as PDF / DOCX / PNG."""
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")

    text = _sessions[session_id]["text"]
    if not text.strip():
        raise HTTPException(400, "No text to export")

    session_dir = _get_session_dir(session_id)
    fmt = fmt.lower()

    try:
        if fmt == "pdf":
            out = export_as_pdf(text, str(session_dir / "output.pdf"))
            return FileResponse(out, media_type="application/pdf",
                                filename="hindi_htr_output.pdf")
        elif fmt == "docx":
            out = export_as_docx(text, str(session_dir / "output.docx"))
            return FileResponse(out,
                                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                filename="hindi_htr_output.docx")
        elif fmt == "png":
            if not _sessions[session_id]["is_single_page"]:
                raise HTTPException(400, "PNG export only available for single-page uploads")
            out = export_as_png(text, str(session_dir / "output.png"))
            return FileResponse(out, media_type="image/png",
                                filename="hindi_htr_output.png")
        else:
            raise HTTPException(400, f"Unsupported format: {fmt}. Use: pdf, docx, png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Export failed: {str(e)}")


# ── GET /api/download/audio/{format} ──────────────────────────
@app.get("/api/download/audio/{fmt}/{session_id}/{voice}")
async def download_audio(fmt: str, session_id: str, voice: str):
    """Download audio as WAV / MP3 / OGG."""
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found")

    wav_path = _sessions[session_id]["audio_paths"].get(voice)
    if not wav_path or not Path(wav_path).exists():
        raise HTTPException(404, f"No {voice} audio generated yet")

    session_dir = _get_session_dir(session_id)
    fmt = fmt.lower()

    if fmt == "wav":
        return FileResponse(wav_path, media_type="audio/wav",
                            filename=f"hindi_htr_{voice}.wav")

    if fmt not in ("mp3", "ogg"):
        raise HTTPException(400, f"Unsupported format: {fmt}. Use: wav, mp3, ogg")

    out_path = str(session_dir / f"audio_{voice}.{fmt}")
    try:
        convert_audio(wav_path, fmt, out_path)
    except Exception as e:
        raise HTTPException(500, f"Audio conversion failed: {str(e)}")

    media_types = {"mp3": "audio/mpeg", "ogg": "audio/ogg"}
    return FileResponse(out_path, media_type=media_types[fmt],
                        filename=f"hindi_htr_{voice}.{fmt}")