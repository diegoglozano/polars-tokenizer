"""Dependency-free error summaries for future raw-text token estimators.

This module scores supplied predictions; it does not fit an estimator or
provide a representative evaluation corpus.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Prediction:
    """One raw-text estimate and its exact reference count."""

    text: str
    actual_tokens: int
    estimated_tokens: float
    language: str
    content_type: str


def utf8_length_class(text: str) -> str:
    """Bucket text by UTF-8 bytes using the benchmark length boundaries."""
    byte_length = len(text.encode())
    if byte_length <= 24:
        return "tiny"
    if byte_length <= 128:
        return "short"
    if byte_length <= 2_048:
        return "medium"
    return "long"


def token_count_class(actual_tokens: int) -> str:
    """Bucket exact counts without using the estimator's prediction."""
    if isinstance(actual_tokens, bool) or not isinstance(actual_tokens, int) or actual_tokens < 0:
        raise ValueError("actual_tokens must be a nonnegative integer")
    if actual_tokens == 0:
        return "0"
    if actual_tokens <= 8:
        return "1-8"
    if actual_tokens <= 32:
        return "9-32"
    if actual_tokens <= 128:
        return "33-128"
    return "129+"


def _nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _validate(rows: list[Prediction]) -> None:
    if not rows:
        raise ValueError("at least one prediction is required")
    for row in rows:
        if not isinstance(row.text, str):
            raise ValueError("text must be a string")
        if isinstance(row.actual_tokens, bool) or not isinstance(row.actual_tokens, int):
            raise ValueError("actual_tokens must be a nonnegative integer")
        if row.actual_tokens < 0:
            raise ValueError("actual_tokens must be a nonnegative integer")
        if (
            isinstance(row.estimated_tokens, bool)
            or not isinstance(row.estimated_tokens, (int, float))
            or not math.isfinite(row.estimated_tokens)
            or row.estimated_tokens < 0
        ):
            raise ValueError("estimated_tokens must be finite and nonnegative")
        if (
            not isinstance(row.language, str)
            or not row.language
            or not isinstance(row.content_type, str)
            or not row.content_type
        ):
            raise ValueError("language and content_type labels must be nonempty")


def _metrics(rows: list[Prediction]) -> dict[str, int | float | None]:
    absolute = [abs(row.estimated_tokens - row.actual_tokens) for row in rows]
    signed = [row.estimated_tokens - row.actual_tokens for row in rows]
    relative = [
        error / row.actual_tokens * 100
        for row, error in zip(rows, absolute, strict=True)
        if row.actual_tokens > 0
    ]
    return {
        "samples": len(rows),
        "zero_actual_samples": len(rows) - len(relative),
        "mae_tokens": statistics.fmean(absolute),
        "median_absolute_error_tokens": statistics.median(absolute),
        "bias_tokens": statistics.fmean(signed),
        "mape_percent": statistics.fmean(relative) if relative else None,
        "p50_relative_error_percent": _nearest_rank(relative, 0.50) if relative else None,
        "p95_relative_error_percent": _nearest_rank(relative, 0.95) if relative else None,
        "p99_relative_error_percent": _nearest_rank(relative, 0.99) if relative else None,
        "max_relative_error_percent": max(relative) if relative else None,
    }


def score_predictions(rows: list[Prediction]) -> dict[str, Any]:
    """Score overall and by language, content, byte length, count, and ASCII class.

    Zero-token references contribute to absolute-error and bias metrics but
    are excluded from percentage-error metrics, whose denominator is zero.
    P50/P95/P99 use nearest-rank percentiles over nonzero-reference rows.
    """
    _validate(rows)
    dimensions: dict[str, dict[str, list[Prediction]]] = {
        name: defaultdict(list)
        for name in (
            "language",
            "content_type",
            "utf8_length_class",
            "token_count_class",
            "ascii_class",
        )
    }
    for row in rows:
        dimensions["language"][row.language].append(row)
        dimensions["content_type"][row.content_type].append(row)
        dimensions["utf8_length_class"][utf8_length_class(row.text)].append(row)
        dimensions["token_count_class"][token_count_class(row.actual_tokens)].append(row)
        dimensions["ascii_class"]["ascii" if row.text.isascii() else "non_ascii"].append(row)

    return {
        "schema_version": 1,
        "overall": _metrics(rows),
        "by": {
            dimension: {label: _metrics(group) for label, group in sorted(groups.items())}
            for dimension, groups in dimensions.items()
        },
    }
