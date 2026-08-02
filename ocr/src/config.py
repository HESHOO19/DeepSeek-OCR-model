"""Application settings for the OCR service.

NOTE on generation_params below: these values are not read by
local_deepseek_ocr/server.py. DeepSeek-OCR's own infer() method doesn't
accept temperature/top_p/top_k/repetition_penalty as arguments at all (its
signature is tokenizer, prompt, image_file, output_path, base_size,
image_size, crop_mode, test_compress, save_results, eval_mode only), so this
dict was never actually reaching the model regardless of what's set here or
in the request payload -- confirmed against the real source of
modeling_deepseekocr.py. The knobs that do reach decoding for the local
server live in local_deepseek_ocr/server.py as DEEPSEEK_NO_REPEAT_NGRAM /
DEEPSEEK_REPETITION_PENALTY / DEEPSEEK_NUM_BEAMS env vars instead, via a
generate() wrapper there. This dict is kept here only in case OCR_API_URL is
pointed at a real OpenAI-compatible / vLLM endpoint instead of the local
server, where these fields are meaningful.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field


_DEFAULT_GENERATION_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repetition_penalty": 1.03,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "min_p": 0.0,
}

_GENERATION_PARAM_BOUNDS = {
    "temperature": (0.0, 1.0),
    "top_p": (0.0, 1.0),
    "top_k": (1, 200),
    "repetition_penalty": (1.0, 2.0),
    "frequency_penalty": (0.0, 2.0),
    "presence_penalty": (0.0, 2.0),
    "min_p": (0.0, 1.0),
}


def _candidate_generation_param_paths() -> list[Path]:
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent

    candidates: list[Path] = []
    env_path = os.getenv("OCR_GENERATION_PARAMS_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            src_dir / "generation_params.json",
            project_root / "generation_params.json",
            project_root / "generation_params 1.json",
        ]
    )
    return candidates


def _coerce_generation_param(key: str, value: object) -> float | int:
    default_value = _DEFAULT_GENERATION_PARAMS[key]
    lower, upper = _GENERATION_PARAM_BOUNDS[key]

    try:
        if isinstance(default_value, int):
            coerced: float | int = int(value)
        else:
            coerced = float(value)
    except (TypeError, ValueError):
        coerced = default_value

    if coerced < lower:
        return lower
    if coerced > upper:
        return upper
    return coerced


def _load_generation_params() -> dict[str, float | int]:
    data: dict[str, object] = {}
    for config_path in _candidate_generation_param_paths():
        if not config_path.exists():
            continue

        try:
            with config_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:
            continue

        if isinstance(loaded, dict):
            data = loaded
            break

    params = dict(_DEFAULT_GENERATION_PARAMS)
    for key, default_value in _DEFAULT_GENERATION_PARAMS.items():
        params[key] = _coerce_generation_param(key, data.get(key, default_value))
    return params


class Settings(BaseModel):
    """Application configuration loaded from defaults, .env, and environment."""

    OCR_API_URL: str = Field(
        default="http://localhost:8009/v1/chat/completions",
        description="URL of the Vision-Language model chat completions endpoint",
    )
    OCR_MODEL: str = Field(
        default="deepseek-ai/DeepSeek-OCR",
        description="Model identifier to use for OCR",
    )

    OCR_TIMEOUT: int = Field(
        default=180,
        description="Timeout in seconds for the VL model HTTP request",
    )
    MAX_TOKENS: int = Field(
        default=8192,
        description="Maximum tokens the model can generate in one response",
    )
    MAX_SOFT_TOKENS: int = Field(
        default=2048,
        description="Gemma visual token budget passed to vLLM via mm_processor_kwargs",
    )
    TEMPERATURE: float = Field(
        default=0.0,
        description="Sampling temperature; low values improve OCR accuracy",
    )
    PDF_DPI: int = Field(
        default=250,
        description="DPI used when rendering PDF pages before OCR",
    )

    MAX_FILE_SIZE_MB: int = Field(
        default=20,
        description="Maximum upload file size in megabytes",
    )
    ALLOWED_IMAGE_EXTENSIONS: set[str] = Field(
        default={".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"},
        description="Accepted image file extensions",
    )
    ALLOWED_PDF_EXTENSIONS: set[str] = Field(
        default={".pdf"},
        description="Accepted PDF file extensions",
    )

    APP_TITLE: str = "DMS OCR Service"
    APP_VERSION: str = "1.0.0"
    BACKEND_PORT: str = Field(
        default="4025",
        description="Port the backend will bind to",
    )

    generation_params: dict[str, float | int] = Field(default_factory=_load_generation_params)

    @property
    def max_file_size_bytes(self) -> int:
        """Return the max file size in bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    @property
    def all_allowed_extensions(self) -> set[str]:
        """Return all accepted upload extensions."""
        return self.ALLOWED_IMAGE_EXTENSIONS | self.ALLOWED_PDF_EXTENSIONS

    @property
    def backend_port(self) -> int:
        """Return the backend port as an integer, with a safe fallback."""
        try:
            return int(self.BACKEND_PORT)
        except (TypeError, ValueError):
            return 4025


def _load_dotenv() -> dict[str, str]:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _coerce_env_value(value: str, target_type: type) -> object:
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is set:
        return {item.strip() for item in value.split(",") if item.strip()}
    if target_type is dict:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object.")
        return parsed
    return value


def _build_settings() -> Settings:
    dotenv_values = _load_dotenv()
    overrides: dict[str, object] = {}

    for name, field_info in Settings.model_fields.items():
        raw_value = os.getenv(name, dotenv_values.get(name))
        if raw_value is None:
            continue

        annotation = field_info.annotation
        origin = getattr(annotation, "__origin__", None)
        if origin is set:
            target_type = set
        elif origin is dict:
            target_type = dict
        else:
            target_type = annotation

        try:
            overrides[name] = _coerce_env_value(raw_value, target_type)
        except (TypeError, ValueError):
            continue

    return Settings(**overrides)


settings = _build_settings()
