import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple


class OCRService:
    """Extract text for policy-document ingestion.

    In the current architecture this service is used by the policy RAG indexer.
    PDF extraction is page-aware: born-digital pages use native text, while
    scanned or text-poor policy pages fall back to Tesseract. This allows hybrid
    policy PDFs to be indexed with page-level provenance.
    """

    MIN_NATIVE_TEXT_CHARS = 40
    MIN_NATIVE_ALNUM_CHARS = 20

    def __init__(self, documents_dir: Path):
        self.documents_dir = documents_dir

    @staticmethod
    def get_supported_extensions() -> List[str]:
        return [".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    @staticmethod
    def tesseract_available() -> bool:
        return shutil.which("tesseract") is not None

    def _run_tesseract_on_image(self, image) -> str:
        import pytesseract

        try:
            return pytesseract.image_to_string(image, lang="spa+eng").strip()
        except Exception:
            return pytesseract.image_to_string(image).strip()

    @staticmethod
    def _read_txt(path: Path) -> Dict[str, Any]:
        return {
            "status": "read",
            "text": path.read_text(encoding="utf-8"),
            "extraction_method": "plain_text",
        }

    def _read_image(self, path: Path) -> Dict[str, Any]:
        if not self.tesseract_available():
            return {
                "status": "error",
                "text": "",
                "extraction_method": "tesseract_ocr",
                "error": "Tesseract executable not found in PATH.",
            }
        try:
            from PIL import Image

            text = self._run_tesseract_on_image(Image.open(path))
            return {
                "status": "read" if text else "warning",
                "text": text,
                "extraction_method": "tesseract_ocr",
                "warning": None if text else "No text extracted.",
            }
        except Exception as exc:
            return {
                "status": "error",
                "text": "",
                "extraction_method": "tesseract_ocr",
                "error": str(exc),
            }

    @classmethod
    def _native_text_is_sufficient(cls, text: str) -> bool:
        cleaned = (text or "").strip()
        if len(cleaned) < cls.MIN_NATIVE_TEXT_CHARS:
            return False
        alnum = sum(char.isalnum() for char in cleaned)
        return alnum >= cls.MIN_NATIVE_ALNUM_CHARS

    def _ocr_fitz_page(self, page) -> str:
        if not self.tesseract_available():
            return ""
        from PIL import Image

        pixmap = page.get_pixmap(matrix=__import__("fitz").Matrix(2.0, 2.0), alpha=False)
        image = Image.open(BytesIO(pixmap.tobytes("png")))
        return self._run_tesseract_on_image(image)

    @staticmethod
    def _format_page_text(page_number: int, text: str) -> str:
        # Page markers remain in the indexed text, making RAG evidence easier
        # to trace back to the originating PDF page.
        return f"[Página {page_number}]\n{text.strip()}".strip()

    def _extract_pdf_pagewise(self, path: Path) -> Tuple[str, List[Dict[str, Any]], List[int]]:
        import fitz

        doc = fitz.open(str(path))
        page_results: List[Dict[str, Any]] = []
        page_texts: List[str] = []
        unreadable_pages: List[int] = []

        try:
            for index, page in enumerate(doc):
                page_number = index + 1
                native_text = (page.get_text("text") or "").strip()
                selected_text = native_text
                method = "native"
                ocr_attempted = False

                if not self._native_text_is_sufficient(native_text):
                    ocr_attempted = self.tesseract_available()
                    ocr_text = ""
                    if ocr_attempted:
                        try:
                            ocr_text = (self._ocr_fitz_page(page) or "").strip()
                        except Exception:
                            ocr_text = ""

                    # Keep the richest extraction. A short native header/page
                    # number should not prevent OCR of the scanned page body.
                    if ocr_text and len(ocr_text) > len(native_text):
                        selected_text = ocr_text
                        method = "ocr"
                    elif native_text:
                        selected_text = native_text
                        method = "native_low_text"
                    elif ocr_text:
                        selected_text = ocr_text
                        method = "ocr"
                    else:
                        selected_text = ""
                        method = "unreadable"
                        unreadable_pages.append(page_number)

                if selected_text:
                    page_texts.append(self._format_page_text(page_number, selected_text))

                page_results.append(
                    {
                        "page": page_number,
                        "method": method,
                        "native_chars": len(native_text),
                        "extracted_chars": len(selected_text),
                        "ocr_attempted": ocr_attempted,
                    }
                )
        finally:
            doc.close()

        return "\n\n".join(page_texts).strip(), page_results, unreadable_pages

    @staticmethod
    def _extract_pdf_text_pypdf(path: Path) -> str:
        """Last-resort parser if PyMuPDF cannot open/process the PDF."""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"[Página {index}]\n{text}")
        return "\n\n".join(pages).strip()

    def _read_pdf(self, path: Path) -> Dict[str, Any]:
        try:
            try:
                text, page_extraction, unreadable_pages = self._extract_pdf_pagewise(path)
            except Exception:
                # Keep compatibility with unusual PDFs that pypdf can read even
                # when PyMuPDF fails. OCR page metadata is unavailable here.
                text = self._extract_pdf_text_pypdf(path)
                if text:
                    return {
                        "status": "read",
                        "text": text,
                        "extraction_method": "pdf_text_extraction_pypdf",
                        "pdf_ocr_fallback_used": False,
                        "page_extraction": [],
                        "ocr_pages": [],
                        "native_pages": [],
                        "unreadable_pages": [],
                    }
                raise

            ocr_pages = [row["page"] for row in page_extraction if row["method"] == "ocr"]
            native_pages = [
                row["page"]
                for row in page_extraction
                if row["method"] in {"native", "native_low_text"}
            ]

            if ocr_pages and native_pages:
                extraction_method = "pdf_hybrid_text_ocr"
            elif ocr_pages:
                extraction_method = "pdf_ocr_tesseract"
            else:
                extraction_method = "pdf_text_extraction"

            if text:
                result: Dict[str, Any] = {
                    "status": "read",
                    "text": text,
                    "extraction_method": extraction_method,
                    "pdf_ocr_fallback_used": bool(ocr_pages),
                    "page_extraction": page_extraction,
                    "ocr_pages": ocr_pages,
                    "native_pages": native_pages,
                    "unreadable_pages": unreadable_pages,
                }
                if unreadable_pages:
                    result["warning"] = (
                        "Some PDF pages produced no usable text: "
                        + ", ".join(map(str, unreadable_pages))
                    )
                return result

            reason = "PDF has no usable selectable text and OCR produced no text."
            if not self.tesseract_available():
                reason = "PDF has no usable selectable text and Tesseract is not available in PATH."
            return {
                "status": "warning",
                "text": "",
                "extraction_method": "pdf_ocr_tesseract",
                "pdf_ocr_fallback_used": True,
                "page_extraction": page_extraction,
                "ocr_pages": ocr_pages,
                "native_pages": native_pages,
                "unreadable_pages": unreadable_pages,
                "warning": reason,
            }
        except Exception as exc:
            return {
                "status": "error",
                "text": "",
                "extraction_method": "pdf_processing_error",
                "error": str(exc),
            }

    def read_document(self, document_name: str) -> Dict[str, Any]:
        safe_name = Path(document_name).name
        path = self.documents_dir / safe_name
        if not path.exists():
            return {
                "document_name": safe_name,
                "status": "not_found",
                "text": "",
                "extraction_method": "not_available",
            }
        if path.suffix.lower() not in self.get_supported_extensions():
            return {
                "document_name": safe_name,
                "status": "unsupported",
                "text": "",
                "extraction_method": "unsupported_format",
            }
        if path.suffix.lower() == ".txt":
            result = self._read_txt(path)
        elif path.suffix.lower() == ".pdf":
            result = self._read_pdf(path)
        else:
            result = self._read_image(path)
        return {"document_name": safe_name, **result}
