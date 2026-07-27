"""PDF to per-page PNG conversion, using PyMuPDF."""

from __future__ import annotations

import io

from PIL import Image


def pdf_to_images(pdf_bytes: bytes, *, dpi: int = 250) -> list[bytes]:
    """Convert a PDF into one PNG byte payload per page, in page order."""
    try:
        import fitz
    except ImportError as exc:
        raise ValueError(
            "PDF support requires PyMuPDF. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open PDF: {exc}") from exc

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    page_images: list[bytes] = []
    try:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            page_images.append(buffer.getvalue())
    finally:
        doc.close()

    if not page_images:
        raise ValueError("PDF was opened but contains no renderable pages.")

    return page_images
