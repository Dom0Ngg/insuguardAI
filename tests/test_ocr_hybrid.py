from io import BytesIO
from pathlib import Path

import pytest

from backend.ocr.ocr_service import OCRService


fitz = pytest.importorskip("fitz")
PIL = pytest.importorskip("PIL.Image")


def _png_with_text(text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1800, 500), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", 48)
    draw.text((80, 160), text, fill="black", font=font)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_hybrid_pdf(path: Path) -> None:
    doc = fitz.open()
    page1 = doc.new_page(width=595, height=842)
    page1.insert_text((70, 100), "POLIZA DIGITAL CON TEXTO SELECCIONABLE Y COBERTURA DE COLISION", fontsize=14)

    page2 = doc.new_page(width=595, height=842)
    image = _png_with_text("CLAUSULA ESCANEADA ROBO DEL VEHICULO CUBIERTO")
    page2.insert_image(fitz.Rect(30, 180, 565, 420), stream=image)
    doc.save(str(path))
    doc.close()


def test_hybrid_pdf_uses_native_text_and_ocr_per_page(tmp_path: Path):
    if not OCRService.tesseract_available():
        pytest.skip("Tesseract is not available")

    pdf = tmp_path / "hybrid.pdf"
    _make_hybrid_pdf(pdf)

    result = OCRService(tmp_path).read_document(pdf.name)

    assert result["status"] == "read"
    assert result["extraction_method"] == "pdf_hybrid_text_ocr"
    assert result["pdf_ocr_fallback_used"] is True
    assert result["native_pages"] == [1]
    assert result["ocr_pages"] == [2]
    assert "POLIZA DIGITAL" in result["text"]
    assert "ROBO" in result["text"].upper()
    assert "[Página 1]" in result["text"]
    assert "[Página 2]" in result["text"]


def test_born_digital_pdf_does_not_require_ocr(tmp_path: Path):
    pdf = tmp_path / "digital.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((70, 100), "DOCUMENTO DIGITAL CON TEXTO SUFICIENTE PARA EXTRACCION NATIVA " * 2)
    doc.save(str(pdf))
    doc.close()

    result = OCRService(tmp_path).read_document(pdf.name)

    assert result["status"] == "read"
    assert result["extraction_method"] == "pdf_text_extraction"
    assert result["pdf_ocr_fallback_used"] is False
    assert result["native_pages"] == [1]
    assert result["ocr_pages"] == []
