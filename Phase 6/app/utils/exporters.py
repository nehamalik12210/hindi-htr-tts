"""
Phase 6 — Text exporters: recognized text → PDF / DOCX / PNG.
Fix 3 applied: uses .ttf font for backend Hindi rendering.
Exports ONLY the converted text — no headers.
"""

import io
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


FONT_DIR = Path(__file__).parent / "fonts"
FONT_PATH = FONT_DIR / "NotoSansDevanagari-Regular.ttf"


def _get_font(size: int = 16):
    """Load Devanagari TTF font, fall back to default if not found."""
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default()


def export_as_pdf(text: str, output_path: str) -> str:
    """Generate a PDF with just the converted Hindi text — no header."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=25)

    if FONT_PATH.exists():
        pdf.add_font("NotoDevnagari", "", str(FONT_PATH), uni=True)
        pdf.set_font("NotoDevnagari", size=14)
    else:
        pdf.set_font("Helvetica", size=14)

    # Just the text — no title/header
    pdf.multi_cell(0, 8, text)

    pdf.output(output_path)
    return output_path


def export_as_docx(text: str, output_path: str) -> str:
    """Generate a DOCX with just the converted Hindi text — no header."""
    from docx import Document
    from docx.shared import Pt

    doc = Document()

    for line in text.split("\n"):
        if line.strip():
            p = doc.add_paragraph(line.strip())
            for run in p.runs:
                run.font.size = Pt(14)
                run.font.name = "Noto Sans Devanagari"

    doc.save(output_path)
    return output_path


def export_as_png(text: str, output_path: str, max_width: int = 800) -> str:
    """Render just the converted Hindi text onto a PNG — no header."""
    font = _get_font(size=24)
    padding = 40
    line_height = 36

    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current_line = words[0]
        for word in words[1:]:
            test = current_line + " " + word
            bbox = font.getbbox(test)
            if bbox[2] - bbox[0] > max_width - 2 * padding:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test
        lines.append(current_line)

    img_height = max(200, padding * 2 + len(lines) * line_height)
    img = Image.new("RGB", (max_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(30, 30, 30), font=font)
        y += line_height

    img.save(output_path)
    return output_path
