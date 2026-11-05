"""OCR fallback for scanned / image-only PDF pages (no system install needed)."""
import numpy as np

_ENGINE = None


def _fitz():
    try:
        import pymupdf as fitz   # newer PyMuPDF
    except ImportError:
        import fitz              # older PyMuPDF
    return fitz


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR
        _ENGINE = RapidOCR()
    return _ENGINE


def _page_to_image(page, zoom: float = 2.0):
    fitz = _fitz()
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    if img.mean() < 110:          # dark background (e.g. dark-mode notes) -> invert
        img = 255 - img
    return np.stack([img, img, img], axis=-1)


def ocr_pages(pdf_path: str, page_indexes) -> dict:
    """Return {page_index: text} for the given 0-based pages."""
    try:
        fitz = _fitz()
        engine = _get_engine()
    except ImportError as e:
        raise ValueError(
            "This PDF has pages with no text (scanned images) and the OCR packages are "
            "missing. Run: pip install pymupdf rapidocr-onnxruntime"
        ) from e

    out = {}
    with fitz.open(pdf_path) as pdf:
        for i in page_indexes:
            result, _ = engine(_page_to_image(pdf[i]))
            lines = []
            for item in (result or []):
                try:
                    if float(item[2]) >= 0.3:
                        lines.append(str(item[1]))
                except (ValueError, IndexError, TypeError):
                    continue
            out[i] = "\n".join(lines)
    return out
