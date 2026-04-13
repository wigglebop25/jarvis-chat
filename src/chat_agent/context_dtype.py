"""
Context dtype compatibility and conversion utilities.

This module handles dtype validation and deterministic conversion between
numeric context payload formats (fp32/fp16/fp8 simulation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import struct
import time


SUPPORTED_CONTEXT_DTYPES = {"fp32", "fp16", "fp8"}


class ContextDTypeError(ValueError):
    """Raised when an unsupported context dtype is requested."""


@dataclass(slots=True)
class DTypeConversionResult:
    source_dtype: str
    target_dtype: str
    values: list[float]
    conversion_ms: float
    size_before_bytes: int
    size_after_bytes: int
    scale: float | None = None
    applied: bool = True
    note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def validate_context_dtype(dtype: str) -> str:
    normalized = dtype.lower().strip()
    if normalized not in SUPPORTED_CONTEXT_DTYPES:
        raise ContextDTypeError(
            f"Unsupported context dtype '{dtype}'. "
            f"Supported dtypes: {sorted(SUPPORTED_CONTEXT_DTYPES)}"
        )
    return normalized


def dtype_size_bytes(dtype: str) -> int:
    normalized = validate_context_dtype(dtype)
    if normalized == "fp32":
        return 4
    if normalized == "fp16":
        return 2
    return 1


def estimate_memory_bytes(value_count: int, dtype: str) -> int:
    return max(0, value_count) * dtype_size_bytes(dtype)


def compatible_dtypes_for_model(provider: str, model: str) -> set[str]:
    """
    Return supported context dtypes for a provider/model profile.

    This is a conservative compatibility guard for context artifacts.
    """
    provider_key = provider.lower().strip()
    model_key = model.lower().strip()

    if provider_key in {"gemini", "google"}:
        return {"fp32", "fp16", "fp8"}
    if "gemma" in model_key:
        return {"fp32", "fp16", "fp8"}
    if provider_key in {"openai", "copilot"}:
        return {"fp32", "fp16"}
    if provider_key == "ollama":
        return {"fp32", "fp16"}
    return {"fp32", "fp16"}


def resolve_target_dtype(preferred_dtype: str, compatible_dtypes: set[str]) -> str:
    preferred = validate_context_dtype(preferred_dtype)
    if preferred in compatible_dtypes:
        return preferred
    for fallback in ("fp16", "fp32", "fp8"):
        if fallback in compatible_dtypes:
            return fallback
    return "fp32"


def _to_fp16(values: list[float]) -> list[float]:
    converted: list[float] = []
    for value in values:
        # Pack/unpack with IEEE 754 half precision.
        half_bytes = struct.pack("<e", float(value))
        converted.append(struct.unpack("<e", half_bytes)[0])
    return converted


def _to_fp8(values: list[float]) -> tuple[list[float], float]:
    """
    Simulate fp8 conversion with deterministic linear quantization.

    We quantize to signed 8-bit range and dequantize back to float values.
    """
    if not values:
        return [], 1.0

    max_abs = max(abs(v) for v in values)
    scale = max_abs / 127.0 if max_abs > 0 else 1.0
    quantized = [max(-127, min(127, int(round(v / scale)))) for v in values]
    dequantized = [q * scale for q in quantized]
    return dequantized, scale


def convert_dtype(
    values: list[float],
    source_dtype: str,
    target_dtype: str,
) -> DTypeConversionResult:
    source = validate_context_dtype(source_dtype)
    target = validate_context_dtype(target_dtype)
    value_count = len(values)
    before_bytes = estimate_memory_bytes(value_count, source)
    start = time.perf_counter()

    numeric_values = [float(v) for v in values]
    if source == target:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return DTypeConversionResult(
            source_dtype=source,
            target_dtype=target,
            values=numeric_values,
            conversion_ms=elapsed_ms,
            size_before_bytes=before_bytes,
            size_after_bytes=before_bytes,
            applied=False,
            note="dtype already matched",
        )

    # Normalize source to fp32-like representation.
    if source == "fp16":
        normalized = _to_fp16(numeric_values)
    elif source == "fp8":
        normalized, _ = _to_fp8(numeric_values)
    else:
        normalized = numeric_values

    scale: float | None = None
    if target == "fp32":
        converted = normalized
    elif target == "fp16":
        converted = _to_fp16(normalized)
    else:
        converted, scale = _to_fp8(normalized)

    after_bytes = estimate_memory_bytes(value_count, target)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return DTypeConversionResult(
        source_dtype=source,
        target_dtype=target,
        values=converted,
        conversion_ms=elapsed_ms,
        size_before_bytes=before_bytes,
        size_after_bytes=after_bytes,
        scale=scale,
    )
