"""Benchmark Google's local Gemma 3 paths without adding project dependencies."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from statistics import median
from typing import Any, Final, TypeVar

from benchmarks._data import (
    CONTENT_TYPES,
    LENGTH_BYTES,
    dataset_digest,
    dataset_statistics,
    make_dataset,
)

T = TypeVar("T")

IMPLEMENTATIONS: Final = (
    "google_local_scalar",
    "google_local_batch_total",
    "sentencepiece_scalar_ids",
    "sentencepiece_batch_ids",
)


@dataclass(frozen=True)
class CountResult:
    total: int
    counts: list[int] | None = None


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


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def machine_metadata() -> dict[str, Any]:
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
        "google_genai": installed_version("google-genai"),
        "sentencepiece": installed_version("sentencepiece"),
        "protobuf": installed_version("protobuf"),
        "git_commit": git.stdout.strip() or None,
        "git_dirty": bool(git_status.stdout.strip()),
    }


def load_google_modules() -> tuple[Any, Any, float]:
    started = time.perf_counter()
    try:
        local_tokenizer = importlib.import_module("google.genai.local_tokenizer")
        loader = importlib.import_module("google.genai._local_tokenizer_loader")
    except ImportError as error:
        raise RuntimeError(
            "Gemini benchmark dependencies are missing. Use the documented uv command."
        ) from error
    return local_tokenizer, loader, time.perf_counter() - started


def artifact_metadata(loader: Any, tokenizer_name: str) -> dict[str, Any]:
    config = getattr(loader, "_TOKENIZERS", {}).get(tokenizer_name)
    if config is None:
        raise ValueError(
            f"{tokenizer_name!r} is not a SentencePiece-backed tokenizer in this SDK version"
        )
    model_url = str(config.model_url)
    cache_path = (
        Path(tempfile.gettempdir())
        / "vertexai_tokenizer_model"
        / hashlib.sha1(model_url.encode()).hexdigest()
    )
    return {
        "tokenizer": tokenizer_name,
        "model_url": model_url,
        "model_sha256": str(config.model_hash),
        "cache_path": str(cache_path),
        "cache_present_before_initialization": cache_path.is_file(),
    }


def google_scalar(tokenizer: Any, texts: Sequence[str]) -> CountResult:
    counts = []
    for text in texts:
        count = tokenizer.count_tokens(text).total_tokens
        if count is None:
            raise RuntimeError("Google LocalTokenizer returned no total token count")
        counts.append(int(count))
    return CountResult(total=sum(counts), counts=counts)


def google_batch_total(tokenizer: Any, texts: Sequence[str]) -> CountResult:
    count = tokenizer.count_tokens(list(texts)).total_tokens
    if count is None:
        raise RuntimeError("Google LocalTokenizer returned no total token count")
    return CountResult(total=int(count))


def sentencepiece_scalar_ids(processor: Any, texts: Sequence[str]) -> CountResult:
    counts = [len(processor.encode(text)) for text in texts]
    return CountResult(total=sum(counts), counts=counts)


def sentencepiece_batch_ids(processor: Any, texts: Sequence[str]) -> CountResult:
    counts = [len(token_ids) for token_ids in processor.encode(list(texts))]
    return CountResult(total=sum(counts), counts=counts)


def implementations(
    tokenizer: Any, processor: Any
) -> dict[str, Callable[[Sequence[str]], CountResult]]:
    return {
        "google_local_scalar": lambda texts: google_scalar(tokenizer, texts),
        "google_local_batch_total": lambda texts: google_batch_total(tokenizer, texts),
        "sentencepiece_scalar_ids": lambda texts: sentencepiece_scalar_ids(processor, texts),
        "sentencepiece_batch_ids": lambda texts: sentencepiece_batch_ids(processor, texts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--length", choices=LENGTH_BYTES, default="short")
    parser.add_argument("--content", choices=CONTENT_TYPES, default="mixed")
    parser.add_argument("--cardinality", type=float, default=1.0)
    parser.add_argument("--null-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--implementation", choices=("all", *IMPLEMENTATIONS), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rows <= 0:
        parser.error("--rows must be positive")
    if args.warm_repeats <= 0:
        parser.error("--warm-repeats must be positive")
    if not 0 < args.cardinality <= 1:
        parser.error("--cardinality must be in (0, 1]")
    if not 0 <= args.null_rate < 1:
        parser.error("--null-rate must be in [0, 1)")
    return args


def main() -> None:
    args = parse_args()
    values = make_dataset(
        args.rows,
        args.length,
        args.cardinality,
        args.seed,
        content=args.content,
        null_rate=args.null_rate,
    )
    texts = [value for value in values if value is not None]
    statistics = dataset_statistics(values)
    byte_count = int(statistics["bytes"])

    local_tokenizer, loader, import_seconds = load_google_modules()
    tokenizer_name = str(loader.get_tokenizer_name(args.model))
    artifact = artifact_metadata(loader, tokenizer_name)
    tokenizer, initialization_samples = timed(lambda: local_tokenizer.LocalTokenizer(args.model))
    processor = loader.get_sentencepiece(tokenizer_name)
    artifact["cache_present_after_initialization"] = Path(artifact["cache_path"]).is_file()
    artifact["cache_size_bytes"] = (
        Path(artifact["cache_path"]).stat().st_size
        if artifact["cache_present_after_initialization"]
        else None
    )

    selected = IMPLEMENTATIONS if args.implementation == "all" else (args.implementation,)
    available = implementations(tokenizer, processor)
    measurements = {}
    results = {}
    for name in selected:
        operation = available[name]
        benchmark = partial(operation, texts)
        cold_result, cold_samples = timed(benchmark)
        warm_result, warm_samples = timed(benchmark, repeats=args.warm_repeats)
        if warm_result != cold_result:
            raise RuntimeError(f"{name} produced different cold and warm results")
        results[name] = warm_result
        measurements[name] = {
            "cold": rate(cold_samples, len(texts), byte_count, warm_result.total),
            "warm": rate(warm_samples, len(texts), byte_count, warm_result.total),
            "materializes_token_ids": name.endswith("_ids") or name.startswith("google_local"),
            "returns_per_row_counts": warm_result.counts is not None,
        }

    totals = {result.total for result in results.values()}
    if len(totals) != 1:
        raise RuntimeError(
            f"implementations disagree on total token count: "
            f"{ {name: result.total for name, result in results.items()} }"
        )
    per_row_counts = {
        tuple(result.counts) for result in results.values() if result.counts is not None
    }
    if len(per_row_counts) > 1:
        raise RuntimeError("implementations disagree on per-row token counts")

    report = {
        "schema_version": 1,
        "dataset": {
            **statistics,
            "length_class": args.length,
            "target_bytes_per_value": LENGTH_BYTES[args.length],
            "content": args.content,
            "requested_cardinality": args.cardinality,
            "requested_null_rate": args.null_rate,
            "seed": args.seed,
            "sha256": dataset_digest(values),
        },
        "tokenizer": {"model_alias": args.model, **artifact},
        "measurements": {
            "module_import_seconds": import_seconds,
            "tokenizer_initialization_seconds": initialization_samples[0]["wall_seconds"],
            "implementations": measurements,
            "peak_process_rss_bytes": peak_rss_bytes(),
            "total_tokens": next(iter(totals)),
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
