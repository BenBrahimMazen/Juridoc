"""
PDF text extraction (PyMuPDF/fitz) with an OCR fallback for scanned pages.

Pages that yield little/no text (scanned/image-only PDFs) are rendered to an
image and OCR'd with rapidocr-onnxruntime (French-capable, CPU, no Tesseract).
OCR is lazy and optional: if rapidocr isn't installed, scanned pages simply
yield empty text (graceful degradation) and text PDFs are unaffected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParsedPDF:
    text: str
    first_page_text: str
    n_pages: int
    n_pages_with_text: int


_WS_RE = re.compile(r"[ \t ]+")
_OCR_MIN_CHARS = 20  # a page with < this many chars of text layer is treated as scanned
_OCR = None  # None=not tried, False=unavailable, <engine>=ready


def _clean(line: str) -> str:
    return _WS_RE.sub(" ", line).rstrip()


def _get_ocr():
    """Lazy singleton RapidOCR; None if the dependency is missing/disabled."""
    global _OCR
    if _OCR is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR = RapidOCR()
        except Exception:
            _OCR = False
    return _OCR if _OCR is not False else None


def _ocr_page(page, dpi: int = 200) -> str:
    ocr = _get_ocr()
    if ocr is None:
        return ""
    import numpy as np
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if img.shape[2] >= 3:
        img = img[:, :, :3]
    try:
        result, _elapse = ocr(img)
    except Exception:
        return ""
    return "\n".join((line[1] or "") for line in (result or []))


def _want_ocr() -> bool:
    try:
        import config
        return getattr(config, "OCR_ENABLED", True)
    except Exception:
        return True


def _ocr_cache_path(pdf_path: Path) -> Path | None:
    """Path to a resumable OCR cache for a PDF (None if config unavailable)."""
    try:
        import config
        return config.PROCESSED_DIR / "ocr_cache" / (Path(pdf_path).stem + ".txt")
    except Exception:
        return None


def extract_pdf(pdf_path: Path, force_ocr: bool = False) -> ParsedPDF:
    """Extract text from a PDF. Pages with little/no text layer are OCR'd.

    `force_ocr=True` ignores the (possibly corrupted) text layer entirely and
    OCRs every page — used for PDFs whose embedded text is garbled (e.g. a broken
    ToUnicode CMap that yields Caesar-shifted garbage), so OCR recovers the real
    text from the rendered page image.
    """
    import fitz  # PyMuPDF

    # Prefer a resumable OCR cache if one exists (pre-OCR'd scanned PDFs);
    # decouples the slow OCR from ingestion and survives interruptions.
    cache = _ocr_cache_path(pdf_path)
    if cache is not None and cache.exists() and not force_ocr:
        cached = cache.read_text(encoding="utf-8", errors="ignore")
        cached = re.sub(r"\[\[PAGE \d+\]\]", "\n", cached)
        cached = re.sub(r"\n{3,}", "\n\n", cached).strip()
        if cached:
            return ParsedPDF(
                text=cached, first_page_text=cached[:1200],
                n_pages=0, n_pages_with_text=1,
            )

    doc = fitz.open(pdf_path)
    ocr_enabled = _want_ocr() or force_ocr
    pages: list[str] = []
    first_page = ""
    with_text = 0
    for i, page in enumerate(doc):
        raw = page.get_text("text") or ""
        if ocr_enabled and (force_ocr or len(raw.strip()) < _OCR_MIN_CHARS):
            ocr_raw = _ocr_page(page)
            if force_ocr:
                raw = ocr_raw if ocr_raw.strip() else raw
            elif len(ocr_raw.strip()) > len(raw.strip()):
                raw = ocr_raw
        cleaned = "\n".join(_clean(ln) for ln in raw.splitlines())
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned:
            with_text += 1
        if i == 0:
            first_page = raw
        pages.append(cleaned)
    doc.close()
    full = "\n\n".join(p for p in pages if p)
    return ParsedPDF(
        text=full,
        first_page_text=first_page.strip(),
        n_pages=len(pages),
        n_pages_with_text=with_text,
    )
