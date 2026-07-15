"""
Phase 6 — File handler: PDF/DOCX/image → list of PIL Images.
"""

import io
from pathlib import Path
from PIL import Image
from app.config import ALLOWED_IMAGE_EXTS, ALLOWED_DOC_EXTS


def extract_pages(file_bytes: bytes, filename: str) -> list[Image.Image]:
    """
    Extract page images from an uploaded file.
    Returns a list of PIL RGB Images (one per page).
    """
    ext = Path(filename).suffix.lower()

    if ext in ALLOWED_IMAGE_EXTS:
        return _handle_image(file_bytes)
    elif ext == ".pdf":
        return _handle_pdf(file_bytes)
    elif ext == ".docx":
        return _handle_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _handle_image(file_bytes: bytes) -> list[Image.Image]:
    """Single image → single-page list."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return [img]


def _handle_pdf(file_bytes: bytes) -> list[Image.Image]:
    """PDF → one image per page at 300 DPI."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 300 DPI for good OCR quality
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        pages.append(img)
    doc.close()
    return pages


def _handle_docx(file_bytes: bytes) -> list[Image.Image]:
    """
    DOCX → extract embedded images.
    If no images found, creates a text-rendered image.
    """
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    images = []

    # Try to extract embedded images
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            try:
                img_data = rel.target_part.blob
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
                images.append(img)
            except Exception:
                continue

    # If no images, extract text paragraphs
    if not images:
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if text.strip():
            # Return a simple text indicator — the text will be used directly
            # Create a minimal placeholder image
            img = Image.new("RGB", (800, 100), (255, 255, 255))
            img.info["extracted_text"] = text
            images.append(img)

    return images
