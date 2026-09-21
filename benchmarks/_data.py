"""Deterministic benchmark corpus generation."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from typing import Final

LENGTH_BYTES: Final = {
    "tiny": 24,
    "short": 128,
    "medium": 2_048,
    "long": 32_768,
}

CONTENT_SAMPLES: Final = {
    "english": (
        "The quick brown fox jumps over the lazy dog. ",
        "A customer asked whether the replacement order had shipped. ",
    ),
    "code": (
        "const result = items.map((item) => item.value);\n",
        "fn count(input: &str) -> usize { input.len() }\n",
    ),
    "json": (
        '{"level":"info","status":200,"duration_ms":12}\n',
        '{"event":"checkout","ok":true,"items":[1,2,3]}\n',
    ),
    "logs": (
        "2026-01-15T12:34:56Z INFO request completed status=200 latency_ms=12\n",
        "WARN retrying upstream request attempt=2 backoff_ms=250\n",
    ),
    "urls": (
        "https://example.com/search?q=token+analytics&lang=en ",
        "mailto:support@example.org?subject=Order%20status ",
    ),
    "spanish": (
        "El cliente solicitó información sobre su pedido. ",
        "La actualización estará disponible mañana por la mañana. ",
    ),
    "cjk": (
        "今天的天气很好，我们去散步吧。",
        "お客様のご注文は明日発送されます。",
    ),
    "emoji": (
        "🚀🌍👋🏽✨ ",
        "status: ✅ build: 🧑‍💻 package: 📦 ",
    ),
}
CONTENT_TYPES: Final = ("mixed", *CONTENT_SAMPLES)


def _value_for(sample: str, target_bytes: int, index: int) -> str:
    suffix = f" [{index}]"
    suffix_bytes = suffix.encode()
    budget = max(0, target_bytes - len(suffix_bytes))
    sample_bytes = sample.encode()
    repeated = (sample_bytes * (budget // len(sample_bytes) + 1))[:budget]
    return repeated.decode(errors="ignore") + suffix


def make_dataset(
    rows: int,
    length: str,
    cardinality: float,
    seed: int,
    *,
    content: str = "mixed",
    null_rate: float = 0.0,
) -> list[str | None]:
    """Build a deterministic dataset with guaranteed non-null cardinality."""
    if rows <= 0:
        raise ValueError("rows must be positive")
    if length not in LENGTH_BYTES:
        raise ValueError(f"unknown length class: {length}")
    if content not in CONTENT_TYPES:
        raise ValueError(f"unknown content type: {content}")
    if not 0 < cardinality <= 1:
        raise ValueError("cardinality must be in (0, 1]")
    if not 0 <= null_rate < 1:
        raise ValueError("null_rate must be in [0, 1)")

    rng = random.Random(seed)
    nulls = min(rows - 1, round(rows * null_rate))
    non_null_rows = rows - nulls
    unique = max(1, min(non_null_rows, round(non_null_rows * cardinality)))
    target = LENGTH_BYTES[length]
    samples = (
        tuple(sample for values in CONTENT_SAMPLES.values() for sample in values)
        if content == "mixed"
        else CONTENT_SAMPLES[content]
    )
    dictionary = [
        _value_for(samples[index % len(samples)], target, index) for index in range(unique)
    ]

    # Cycling first guarantees that every dictionary value occurs. Shuffling
    # then avoids an unrealistically periodic access pattern.
    values: list[str | None] = [dictionary[index % unique] for index in range(non_null_rows)]
    values.extend([None] * nulls)
    rng.shuffle(values)
    return values


def dataset_digest(values: Sequence[str | None]) -> str:
    """Hash values without delimiter ambiguity and with nulls distinguished."""
    digest = hashlib.sha256()
    for value in values:
        if value is None:
            digest.update(b"N")
            continue
        encoded = value.encode()
        digest.update(b"S")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def dataset_statistics(
    values: Sequence[str | None], *, unique_values: int | None = None
) -> dict[str, int | float]:
    null_rows = 0
    byte_count = 0
    observed_values = set() if unique_values is None else None
    for value in values:
        if value is None:
            null_rows += 1
            continue
        byte_count += len(value.encode())
        if observed_values is not None:
            observed_values.add(value)
    non_null_rows = len(values) - null_rows
    unique = len(observed_values) if observed_values is not None else unique_values
    assert unique is not None
    return {
        "rows": len(values),
        "null_rows": null_rows,
        "bytes": byte_count,
        "unique_values": unique,
        "actual_cardinality": unique / non_null_rows if non_null_rows else 0.0,
    }
