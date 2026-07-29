"""Pydantic schemas for the OCR service."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OCRType(str, Enum):
    """Prompt template to use for the OCR request."""

    GENERAL = "general"
    TABLE = "table"
    FORM = "form"
    HANDWRITTEN = "handwritten"
    DENSE = "dense"


class OutputFormat(str, Enum):
    """Requested OCR output shape."""

    MARKDOWN = "markdown"
    JSON = "json"
    PLAIN_TEXT = "plain_text"
    EXCEL = "excel"
    CSV = "csv"


class OCRPageResult(BaseModel):
    """OCR result for a single page."""

    page_number: int
    text: str


class OCRMetadata(BaseModel):
    """Metadata attached to every OCRResponse."""

    model: str
    ocr_type: str
    output_format: str
    preprocessed: bool
    total_pages: int
    tokens_used: dict[str, Any] = Field(default_factory=dict)


class OCRResponse(BaseModel):
    """Top-level response returned by OCRService.process_file()."""

    filename: str
    text: str
    pages: list[OCRPageResult]
    metadata: OCRMetadata
    output_path: str | None = None
    output_url: str | None = None
    md_output_path: str | None = None
