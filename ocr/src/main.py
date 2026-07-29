"""FastAPI entrypoint for the DMS OCR Service."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.config import settings
from src.schemas.ocr import OCRType, OutputFormat
from src.services.ocr_service import OCRService, OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_TITLE, version=settings.APP_VERSION)
ocr_service = OCRService()

GUI_PATH = Path(__file__).resolve().parent.parent / "gui" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def gui() -> str:
    """Serve the local same-origin GUI."""
    return GUI_PATH.read_text(encoding="utf-8")


@app.get("/health")
async def health() -> dict:
    """Confirm the app is up and expose the active OCR settings."""
    return {
        "status": "ok",
        "model": settings.OCR_MODEL,
        "api_url": settings.OCR_API_URL,
        "pdf_dpi": settings.PDF_DPI,
        "max_soft_tokens": settings.MAX_SOFT_TOKENS,
        "generation_params": settings.generation_params,
    }


@app.get("/outputs/{filename}")
async def download_output(filename: str) -> FileResponse:
    """Download a saved OCR output file."""
    output_dir = OUTPUT_DIR.resolve()
    output_path = (output_dir / filename).resolve()

    if output_path.parent != output_dir or not output_path.exists() or not output_path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found.")

    return FileResponse(path=str(output_path), filename=output_path.name)


class ConvertRequest(BaseModel):
    filename: str
    target_format: OutputFormat


@app.post("/convert")
async def convert_output(payload: ConvertRequest) -> dict:
    """Convert an already-saved OCR output (json/md/txt) into any other
    format, including Excel -- regardless of what format it was originally
    saved as."""
    try:
        output_path = ocr_service.convert_output(payload.filename, payload.target_format)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    name = Path(output_path).name
    return {"output_path": output_path, "output_url": f"/outputs/{name}", "filename": name}


@app.post("/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    ocr_type: OCRType = Form(OCRType.GENERAL),
    output_format: OutputFormat = Form(OutputFormat.MARKDOWN),
    preprocess: bool = Form(False),
    resize_factor: float = Form(1.0),
    denoise: bool = Form(False),
    enhance_contrast: bool = Form(True),
    sharpen: bool = Form(True),
    binarize: bool = Form(False),
):
    """Upload a single image or PDF and run OCR on it."""
    filename = file.filename or "uploaded_file"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in settings.all_allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(settings.all_allowed_extensions)}",
        )

    if not 0.25 <= resize_factor <= 4.0:
        raise HTTPException(
            status_code=400,
            detail="resize_factor must be between 0.25 and 4.0.",
        )

    file_bytes = await file.read()

    if len(file_bytes) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(file_bytes)} bytes). Max is {settings.MAX_FILE_SIZE_MB} MB.",
        )

    result = await ocr_service.process_file(
        file_bytes=file_bytes,
        filename=filename,
        is_pdf=ext in settings.ALLOWED_PDF_EXTENSIONS,
        ocr_type=ocr_type,
        output_format=output_format,
        preprocess=preprocess,
        resize_factor=resize_factor,
        denoise=denoise,
        enhance_contrast=enhance_contrast,
        sharpen=sharpen,
        binarize=binarize,
    )

    if isinstance(result, dict) and result.get("success") is False:
        raise HTTPException(status_code=502, detail=result.get("error", "OCR failed"))

    return JSONResponse(content=result.model_dump(mode="json"))
