"""Reproducible end-to-end exact-count benchmark with JSON output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import polars as pl
import polars_tokenizer as tokens
import tiktoken

SAMPLES = (
    "The quick brown fox jumps over the lazy dog. ",
    "const result = items.map((item) => item.value);\n",
    '{"level":"info","status":200,"duration_ms":12}\n',
    "https://example.com/search?q=token+analytics&lang=en ",
    "El cliente solicitó información sobre su pedido. ",
    "今天的天气很好，我们去散步吧。",
    "🚀🌍👋🏽✨ ",
)
LENGTH_BYTES = {"tiny": 24, "short": 128, "medium": 2_048, "long": 32_768}
T = TypeVar("T")


def make_dataset(rows: int, length: str, cardinality: float, seed: int) -> list[str]:
    rng = random.Random(seed)
    unique = max(1, min(rows, round(rows * cardinality)))
    target = LENGTH_BYTES[length]
    values: list[str] = []
    for index in range(unique):
        sample = SAMPLES[index % len(SAMPLES)]
        text = (sample * (target // len(sample.encode()) + 1)).encode()[:target]
        values.append(text.decode(errors="ignore") + f" {index}")
    return [values[rng.randrange(unique)] for _ in range(rows)]


def timed(callable_: Callable[[], T], repeats: int = 1) -> tuple[T, float]:
    start = time.perf_counter()
    result = callable_()
    best = time.perf_counter() - start
    for _ in range(repeats - 1):
        start = time.perf_counter()
        candidate = callable_()
        best = min(best, time.perf_counter() - start)
        result = candidate
    return result, best


def rate(seconds: float, rows: int, byte_count: int) -> dict[str, float]:
    return {
        "seconds": seconds,
        "rows_per_second": rows / seconds,
        "mib_per_second": byte_count / seconds / (1024 * 1024),
    }


def machine_metadata() -> dict[str, Any]:
    rustc = shutil.which("rustc")
    rust = (
        subprocess.run(
            [rustc, "--version"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if rustc
        else None
    )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "rust": rust or None,
        "polars": pl.__version__,
        "tiktoken": tiktoken.__version__,
        "polars_max_threads": os.environ.get("POLARS_MAX_THREADS"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--length", choices=LENGTH_BYTES, default="short")
    parser.add_argument("--cardinality", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows <= 0 or not 0 < args.cardinality <= 1:
        parser.error("--rows must be positive and --cardinality must be in (0, 1]")

    values = make_dataset(args.rows, args.length, args.cardinality, args.seed)
    byte_count = sum(len(value.encode()) for value in values)
    digest = hashlib.sha256("\0".join(values).encode()).hexdigest()
    frame = pl.DataFrame({"text": values})
    expression = tokens.count("text").alias("count")

    _, cold_seconds = timed(lambda: frame.select(expression))
    plugin_result, warm_seconds = timed(lambda: frame.select(expression), repeats=5)

    _, init_seconds = timed(lambda: tiktoken.get_encoding("o200k_base"))
    encoding = tiktoken.get_encoding("o200k_base")
    reference, scalar_seconds = timed(
        lambda: [len(encoding.encode(value, disallowed_special=())) for value in values]
    )
    if plugin_result.to_series().to_list() != reference:
        raise RuntimeError("plugin output differs from tiktoken reference")

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    report = {
        "schema_version": 1,
        "dataset": {
            "rows": args.rows,
            "bytes": byte_count,
            "length_class": args.length,
            "cardinality": args.cardinality,
            "seed": args.seed,
            "sha256": digest,
        },
        "measurements": {
            "plugin_cold": rate(cold_seconds, args.rows, byte_count),
            "plugin_warm": rate(warm_seconds, args.rows, byte_count),
            "python_tiktoken_scalar": rate(scalar_seconds, args.rows, byte_count),
            "tiktoken_initialization_seconds": init_seconds,
            "peak_rss_bytes": rss_bytes,
        },
        "environment": machine_metadata(),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
