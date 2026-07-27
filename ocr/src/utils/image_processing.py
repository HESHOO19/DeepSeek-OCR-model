"""Image preparation utilities for OCR."""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


MAX_LONG_SIDE = 3200
AUTO_PREPROCESS_MIN_LONG_SIDE = 1600


def _open_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)

    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        return background

    if image.mode not in ("RGB", "L"):
        return image.convert("RGB")

    return image


def _resize_image(image: Image.Image, resize_factor: float) -> Image.Image:
    if resize_factor <= 0:
        raise ValueError("resize_factor must be greater than 0")

    long_side = max(image.size)
    if resize_factor == 1.0 and long_side < AUTO_PREPROCESS_MIN_LONG_SIDE:
        resize_factor = AUTO_PREPROCESS_MIN_LONG_SIDE / long_side

    if resize_factor == 1.0:
        return image

    target_width = max(1, int(image.width * resize_factor))
    target_height = max(1, int(image.height * resize_factor))

    scaled_long_side = max(target_width, target_height)
    if scaled_long_side > MAX_LONG_SIDE:
        scale_down = MAX_LONG_SIDE / scaled_long_side
        target_width = max(1, int(target_width * scale_down))
        target_height = max(1, int(target_height * scale_down))

    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_image_bytes_to_base64(image_bytes: bytes) -> str:
    """Return a PNG base64 payload without changing visual content."""
    return _encode_png(_open_image(image_bytes))


def preprocess_image(
    image_bytes: bytes,
    *,
    resize_factor: float = 1.0,
    denoise: bool = False,
    enhance_contrast: bool = False,
    sharpen: bool = False,
    binarize: bool = False,
) -> str:
    """
    Apply OCR-oriented preprocessing and return a base64-encoded PNG.

    Order: orientation/mode fix, resize, denoise, contrast, sharpen, binarize.
    Binarize is intentionally last because it is destructive.
    """
    image = _open_image(image_bytes)
    image = _resize_image(image, resize_factor)

    if denoise:
        image = image.filter(ImageFilter.MedianFilter(size=3))

    if enhance_contrast:
        image = ImageOps.autocontrast(image, cutoff=1)
        image = ImageEnhance.Contrast(image).enhance(1.35)

    if sharpen:
        image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=3))

    if binarize:
        grayscale = ImageOps.autocontrast(image.convert("L"), cutoff=1)
        threshold = 180
        image = grayscale.point(lambda pixel: 255 if pixel > threshold else 0)

    return _encode_png(image)
