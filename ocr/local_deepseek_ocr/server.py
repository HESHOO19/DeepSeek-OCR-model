"""
Local DeepSeek-OCR model server.

DeepSeek-OCR (https://huggingface.co/deepseek-ai/DeepSeek-OCR) does not speak
the OpenAI chat-completions protocol natively -- it's called via a custom
`model.infer(...)` method. This wraps it in a small FastAPI server that
exposes a `/v1/chat/completions` endpoint shaped like the OpenAI API, so the
existing OCR backend (`src/services/ocr_service.py`) can call it unmodified --
just point OCR_API_URL in `.env` at this server.

Run with:
    uvicorn local_deepseek_ocr.server:app --port 8009

Then in ocr/.env:
    OCR_API_URL="http://localhost:8009/v1/chat/completions"
    OCR_MODEL="deepseek-ai/DeepSeek-OCR"

Model loads in a background thread (not at import time) so /health responds
immediately instead of hanging while ~6GB of weights download/load -- same
fix pattern used for the "model offline" bug in the DeepSeek-OCR GUI project.
"""

from __future__ import annotations

import base64
import contextlib
import io
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

app = FastAPI(title="DeepSeek-OCR Local Server")

_state: dict = {"status": "loading", "model": None, "tokenizer": None, "error": None}


def _gpu_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def _load_model() -> None:
    """Load tokenizer + model in a background thread.

    Full bf16 weights for this 3B model are ~6GB, which leaves no headroom
    on cards with <=8GB VRAM (e.g. the 4050) once activations are added --
    that's what caused the hard crash. On small cards we load 4-bit
    quantized via bitsandbytes instead, which drops weight memory to ~2GB.
    """
    try:
        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading DeepSeek-OCR tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        vram_gb = _gpu_vram_gb()
        use_4bit = vram_gb and vram_gb < 10
        logger.info("Detected ~%.1fGB VRAM -> use_4bit=%s", vram_gb, use_4bit)

        model = None

        if use_4bit:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            for attn_impl in ("flash_attention_2", "eager"):
                try:
                    logger.info(
                        "Loading DeepSeek-OCR 4-bit with _attn_implementation=%s ...",
                        attn_impl,
                    )
                    model = AutoModel.from_pretrained(
                        MODEL_NAME,
                        _attn_implementation=attn_impl,
                        trust_remote_code=True,
                        use_safetensors=True,
                        quantization_config=quant_config,
                        device_map={"": 0},
                        torch_dtype=torch.float16,
                    )
                    logger.info("Loaded 4-bit with %s attention.", attn_impl)
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "4-bit load failed with %s (%s), trying next option ...",
                        attn_impl,
                        exc,
                    )

            if model is None:
                raise RuntimeError(
                    "Could not load DeepSeek-OCR 4-bit with either attention backend."
                )

            model = model.eval()
            # bitsandbytes only quantizes nn.Linear layers -- non-quantized
            # submodules (this model's SAM/CLIP vision tower in particular)
            # are left in whatever dtype they defaulted to, which can end up
            # fp32 even with torch_dtype set above. That produces a dtype
            # mismatch between text embeddings (bf16) and vision features
            # (fp32) inside the model's forward pass. Casting the full model
            # afterwards is safe here: bnb's Params4bit.to() only moves
            # device for already-quantized weights, it doesn't touch the
            # packed 4-bit storage, so this only affects the non-quantized
            # (vision tower) parameters.
            logger.info("Keeping bitsandbytes model in its loaded dtype.")
        else:
            for attn_impl in ("flash_attention_2", "eager"):
                try:
                    logger.info(
                        "Loading DeepSeek-OCR model with _attn_implementation=%s ...",
                        attn_impl,
                    )
                    model = AutoModel.from_pretrained(
                        MODEL_NAME,
                        _attn_implementation=attn_impl,
                        trust_remote_code=True,
                        use_safetensors=True,
                    )
                    logger.info("Loaded with %s attention.", attn_impl)
                    break
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to load with %s (%s), trying next option ...",
                        attn_impl,
                        exc,
                    )

            if model is None:
                raise RuntimeError(
                    "Could not load DeepSeek-OCR with either flash_attention_2 or eager attention."
                )
            model = model.eval().cuda().half()

        _state["tokenizer"] = tokenizer
        _state["model"] = model
        _state["status"] = "ready"
        logger.info("DeepSeek-OCR is ready.")
    except Exception as exc:  # noqa: BLE001
        logger.exception("DeepSeek-OCR failed to load")
        _state["status"] = "error"
        _state["error"] = str(exc)


threading.Thread(target=_load_model, daemon=True).start()


@app.get("/health")
async def health() -> dict:
    return {"status": _state["status"], "model": MODEL_NAME, "error": _state["error"]}


def _extract_prompt_and_image(messages: list) -> tuple[str, str | None]:
    """Pull the free-text instruction and base64 image out of an OpenAI-style
    chat-completions `messages` payload."""
    text_parts: list[str] = []
    image_b64: str | None = None

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        for part in content or []:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                image_b64 = url.split(",", 1)[1] if "," in url else url

    return "\n".join(text_parts), image_b64


def _build_deepseek_prompt(instruction_text: str) -> str:
    """Map the caller's free-text instruction onto DeepSeek-OCR's expected
    prompt tokens. Table/structured requests get the grounding+markdown
    prompt; everything else gets plain Free OCR."""
    lowered = instruction_text.lower()
    if "table" in lowered or "markdown" in lowered or "form" in lowered:
        return "<image>\n<|grounding|>Convert the document to markdown."
    return "<image>\nFree OCR."


@app.post("/v1/chat/completions")
async def chat_completions(request: dict) -> JSONResponse:
    if _state["status"] != "ready":
        return JSONResponse(
            status_code=503,
            content={"error": {"message": f"Model not ready (status={_state['status']}, error={_state['error']})"}},
        )

    messages = request.get("messages", [])
    instruction_text, image_b64 = _extract_prompt_and_image(messages)

    if not image_b64:
        return JSONResponse(status_code=400, content={"error": {"message": "No image_url found in messages."}})

    image_bytes = base64.b64decode(image_b64)
    prompt = _build_deepseek_prompt(instruction_text)

    with tempfile.TemporaryDirectory() as tmp_dir:
        image_path = os.path.join(tmp_dir, "input.png")
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        # save_results MUST be True: on upstream DeepSeek-OCR, infer() only
        # populates its return value when save_results=True (confirmed via
        # https://github.com/deepseek-ai/DeepSeek-OCR/issues/249 -- without
        # it, infer() effectively returns nothing usable). Setting it False
        # to silence the console dump (as we tried before) breaks the actual
        # OCR result, not just the noise -- that's what caused the empty
        # json/md/txt outputs. So: keep it True, and suppress the console
        # spam separately via redirect_stdout instead.
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            result = _state["model"].infer(
                _state["tokenizer"],
                prompt=prompt,
                image_file=image_path,
                output_path=tmp_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
            )

        if isinstance(result, dict):
            text = result.get("text", "")
        elif isinstance(result, str):
            text = result
        else:
            text = str(result) if result is not None else ""

        # Fallback: some DeepSeek-OCR versions leave the return value empty
        # even with save_results=True and only write the real text to disk.
        # Must happen inside this `with` block -- tmp_dir is deleted on exit.
        if not text.strip():
            for pattern in ("*.mmd", "*.md", "*.txt"):
                matches = sorted(Path(tmp_dir).rglob(pattern))
                if matches:
                    text = matches[0].read_text(encoding="utf-8", errors="ignore")
                    break

    return JSONResponse(
        content={
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }
    )
