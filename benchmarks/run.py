"""Run one reproducible end-to-end exact-count benchmark case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from statistics import median
from typing import Any, TypeVar

from benchmarks._data import (
    CONTENT_TYPES,
    INPUT_DTYPES,
    LENGTH_BYTES,
    dataset_digest,
    dataset_statistics,
    make_dataset,
)

T = TypeVar("T")


def timed(callable_: Callable[[], T], repeats: int = 1) -> tuple[T, list[dict[str, float]]]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    samples = []
    result: T
    for _ in range(repeats):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = callable_()
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
        samples.append(
            {
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "process_cpu_percent": cpu_seconds / wall_seconds * 100,
            }
        )
    return result, samples


def rate(
    samples: list[dict[str, float]], rows: int, byte_count: int, token_count: int
) -> dict[str, Any]:
    seconds = median(sample["wall_seconds"] for sample in samples)
    return {
        "seconds": seconds,
        "best_seconds": min(sample["wall_seconds"] for sample in samples),
        "rows_per_second": rows / seconds,
        "mib_per_second": byte_count / seconds / (1024 * 1024),
        "tokens_per_second": token_count / seconds,
        "process_cpu_percent": median(sample["process_cpu_percent"] for sample in samples),
        "samples": samples,
    }


def count_digest(counts: list[int | None]) -> str:
    digest = hashlib.sha256()
    for count in counts:
        digest.update(b"\x00" if count is None else b"\x01" + count.to_bytes(8, "little"))
    return digest.hexdigest()


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def machine_metadata(polars_threads: int | None) -> dict[str, Any]:
    rustc = shutil.which("rustc")
    rust = (
        subprocess.run(
            [rustc, "--version"], capture_output=True, text=True, check=False
        ).stdout.strip()
        if rustc
        else None
    )
    try:
        plugin_version = package_version("polars-tokenizer")
    except PackageNotFoundError:
        plugin_version = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "rust": rust or None,
        "polars": package_version("polars"),
        "polars_threads": polars_threads,
        "tiktoken": package_version("tiktoken"),
        "polars_tokenizer": plugin_version,
        "polars_max_threads": os.environ.get("POLARS_MAX_THREADS"),
        "rustflags": os.environ.get("RUSTFLAGS"),
        "git_commit": git.stdout.strip() or None,
        "git_dirty": bool(git_status.stdout.strip()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--length", choices=LENGTH_BYTES, default="short")
    parser.add_argument("--content", choices=CONTENT_TYPES, default="mixed")
    parser.add_argument("--input-dtype", choices=INPUT_DTYPES, default="string")
    parser.add_argument("--tokenizer", choices=("o200k_base", "cl100k_base"), default="o200k_base")
    parser.add_argument("--cardinality", type=float, default=1.0)
    parser.add_argument("--null-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--implementation", choices=("all", "plugin", "reference"), default="all")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warm_repeats <= 0:
        parser.error("--warm-repeats must be positive")
    if args.skip_reference and args.implementation != "all":
        parser.error("--skip-reference cannot be combined with --implementation")
    implementation = "plugin" if args.skip_reference else args.implementation

    try:
        values = make_dataset(
            args.rows,
            args.length,
            args.cardinality,
            args.seed,
            content=args.content,
            null_rate=args.null_rate,
        )
    except ValueError as error:
        parser.error(str(error))
    null_rows = min(args.rows - 1, round(args.rows * args.null_rate))
    non_null_rows = args.rows - null_rows
    unique_values = max(1, min(non_null_rows, round(non_null_rows * args.cardinality)))
    statistics = dataset_statistics(values, unique_values=unique_values)
    byte_count = int(statistics["bytes"])
    plugin_cold = None
    plugin_warm = None
    reference_measurement = None
    initialization_seconds = None
    plugin_values = None
    reference = None
    polars_threads = None
    if implementation in ("all", "plugin"):
        import polars as pl
        import polars_tokenizer as tokens

        series = pl.Series("text", values)
        if args.input_dtype == "categorical":
            series = series.cast(pl.Categorical)
        frame = pl.DataFrame(series)
        expression = tokens.count("text", tokenizer=args.tokenizer).alias("count")

        _, cold_samples = timed(lambda: frame.select(expression))
        plugin_result, warm_samples = timed(
            lambda: frame.select(expression), repeats=args.warm_repeats
        )
        plugin_values = plugin_result.to_series().to_list()
        polars_threads = pl.thread_pool_size()
        total_tokens = sum(value for value in plugin_values if value is not None)
        plugin_cold = rate(cold_samples, args.rows, byte_count, total_tokens)
        plugin_warm = rate(warm_samples, args.rows, byte_count, total_tokens)

    if implementation in ("all", "reference"):
        import tiktoken

        encoding, init_samples = timed(lambda: tiktoken.get_encoding(args.tokenizer))
        initialization_seconds = init_samples[0]["wall_seconds"]
        reference, scalar_samples = timed(
            lambda: [
                len(encoding.encode(value, disallowed_special=())) if value is not None else None
                for value in values
            ]
        )
        if plugin_values is not None and plugin_values != reference:
            raise RuntimeError("plugin output differs from tiktoken reference")
        total_tokens = sum(value for value in reference if value is not None)
        reference_measurement = rate(scalar_samples, args.rows, byte_count, total_tokens)

    counts = plugin_values if plugin_values is not None else reference
    assert counts is not None
    rss_bytes = peak_rss_bytes()
    report = {
        "schema_version": 5,
        "implementation": implementation,
        "dataset": {
            **statistics,
            "length_class": args.length,
            "target_bytes_per_value": LENGTH_BYTES[args.length],
            "content": args.content,
            "input_dtype": args.input_dtype,
            "tokenizer": args.tokenizer,
            "requested_cardinality": args.cardinality,
            "requested_null_rate": args.null_rate,
            "null_rate": statistics["null_rows"] / args.rows,
            "seed": args.seed,
            "sha256": dataset_digest(values),
        },
        "measurements": {
            "plugin_cold": plugin_cold,
            "plugin_warm": plugin_warm,
            "python_tiktoken_scalar": reference_measurement,
            "tiktoken_initialization_seconds": initialization_seconds,
            "peak_rss_bytes": rss_bytes,
            "total_tokens": total_tokens,
            "output_sha256": count_digest(counts),
        },
        "environment": machine_metadata(polars_threads),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
