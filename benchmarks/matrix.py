"""Run isolated benchmark cases across a configurable workload matrix."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

import polars as pl

from benchmarks._data import CONTENT_TYPES, INPUT_DTYPES, LENGTH_BYTES

T = TypeVar("T")


def comma_separated(value: str, convert: Callable[[str], T]) -> list[T]:
    try:
        parsed = [convert(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one value")
    return parsed


def flatten_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    dataset = report["dataset"]
    environment = report["environment"]
    rows = []
    for name in ("plugin_cold", "plugin_warm", "python_tiktoken_scalar"):
        measurement = report["measurements"].get(name)
        if measurement is None:
            continue
        rows.append(
            {
                "implementation": name,
                "rows": dataset["rows"],
                "bytes": dataset["bytes"],
                "length_class": dataset["length_class"],
                "content": dataset["content"],
                "input_dtype": dataset["input_dtype"],
                "tokenizer": dataset["tokenizer"],
                "requested_cardinality": dataset["requested_cardinality"],
                "actual_cardinality": dataset["actual_cardinality"],
                "null_rate": dataset["null_rate"],
                "threads": environment["polars_threads"],
                "seconds": measurement["seconds"],
                "rows_per_second": measurement["rows_per_second"],
                "mib_per_second": measurement["mib_per_second"],
                "tokens_per_second": measurement["tokens_per_second"],
                "process_cpu_percent": measurement["process_cpu_percent"],
                "peak_rss_bytes": report["measurements"][
                    "reference_peak_rss_bytes"
                    if name == "python_tiktoken_scalar"
                    else "plugin_peak_rss_bytes"
                ],
                "dataset_sha256": dataset["sha256"],
                "git_commit": environment["git_commit"],
                "git_dirty": environment["git_dirty"],
            }
        )
    return rows


def combine_reports(plugin: dict[str, Any], reference: dict[str, Any] | None) -> dict[str, Any]:
    if plugin["implementation"] != "plugin":
        raise ValueError("expected an isolated plugin report")
    measurements = plugin["measurements"]
    if reference is not None:
        if reference["implementation"] != "reference":
            raise ValueError("expected an isolated reference report")
        if plugin["dataset"] != reference["dataset"]:
            raise ValueError("isolated benchmark datasets differ")
        reference_measurements = reference["measurements"]
        if (
            measurements["output_sha256"] != reference_measurements["output_sha256"]
            or measurements["total_tokens"] != reference_measurements["total_tokens"]
        ):
            raise ValueError("plugin output differs from tiktoken reference")
    else:
        reference_measurements = None

    return {
        "schema_version": 5,
        "implementation": "isolated",
        "dataset": plugin["dataset"],
        "measurements": {
            "plugin_cold": measurements["plugin_cold"],
            "plugin_warm": measurements["plugin_warm"],
            "python_tiktoken_scalar": (
                reference_measurements["python_tiktoken_scalar"]
                if reference_measurements is not None
                else None
            ),
            "tiktoken_initialization_seconds": (
                reference_measurements["tiktoken_initialization_seconds"]
                if reference_measurements is not None
                else None
            ),
            "plugin_peak_rss_bytes": measurements["peak_rss_bytes"],
            "reference_peak_rss_bytes": (
                reference_measurements["peak_rss_bytes"]
                if reference_measurements is not None
                else None
            ),
            "total_tokens": measurements["total_tokens"],
            "output_sha256": measurements["output_sha256"],
        },
        "environment": plugin["environment"],
    }


def run_implementation(
    command: list[str], environment: dict[str, str], implementation: str
) -> dict[str, Any]:
    completed = subprocess.run(
        [*command, "--implementation", implementation],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(completed.stdout)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def positive(values: Iterable[int], option: str, parser: argparse.ArgumentParser) -> None:
    if any(value <= 0 for value in values):
        parser.error(f"{option} values must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=lambda value: comma_separated(value, int), default=[100_000])
    parser.add_argument(
        "--lengths", type=lambda value: comma_separated(value, str), default=["short"]
    )
    parser.add_argument(
        "--cardinalities", type=lambda value: comma_separated(value, float), default=[1.0]
    )
    parser.add_argument(
        "--contents", type=lambda value: comma_separated(value, str), default=["mixed"]
    )
    parser.add_argument(
        "--dtypes", type=lambda value: comma_separated(value, str), default=["string"]
    )
    parser.add_argument(
        "--tokenizers", type=lambda value: comma_separated(value, str), default=["o200k_base"]
    )
    parser.add_argument("--threads", type=lambda value: comma_separated(value, int), default=[1])
    parser.add_argument("--null-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--max-input-mib", type=int, default=512)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--parquet", type=Path)
    args = parser.parse_args()

    positive(args.rows, "--rows", parser)
    positive(args.threads, "--threads", parser)
    if args.warm_repeats <= 0 or args.max_input_mib <= 0:
        parser.error("--warm-repeats and --max-input-mib must be positive")
    if unknown := sorted(set(args.lengths) - set(LENGTH_BYTES)):
        parser.error(f"unknown length classes: {', '.join(unknown)}")
    if unknown := sorted(set(args.contents) - set(CONTENT_TYPES)):
        parser.error(f"unknown content types: {', '.join(unknown)}")
    if unknown := sorted(set(args.dtypes) - set(INPUT_DTYPES)):
        parser.error(f"unknown input dtypes: {', '.join(unknown)}")
    if unknown := sorted(set(args.tokenizers) - {"o200k_base", "cl100k_base"}):
        parser.error(f"unknown tokenizers: {', '.join(unknown)}")
    if any(not 0 < value <= 1 for value in args.cardinalities):
        parser.error("--cardinalities values must be in (0, 1]")
    if not 0 <= args.null_rate < 1:
        parser.error("--null-rate must be in [0, 1)")

    reports = []
    skipped = []
    cases = list(
        itertools.product(
            args.rows,
            args.lengths,
            args.cardinalities,
            args.contents,
            args.dtypes,
            args.tokenizers,
            args.threads,
        )
    )
    max_bytes = args.max_input_mib * 1024 * 1024
    for case_number, (
        rows,
        length,
        cardinality,
        content,
        input_dtype,
        tokenizer,
        threads,
    ) in enumerate(cases, start=1):
        estimated_bytes = rows * LENGTH_BYTES[length]
        case = {
            "rows": rows,
            "length": length,
            "cardinality": cardinality,
            "content": content,
            "input_dtype": input_dtype,
            "tokenizer": tokenizer,
            "threads": threads,
        }
        if estimated_bytes > max_bytes:
            skipped.append({**case, "reason": "estimated input exceeds --max-input-mib"})
            continue

        print(f"[{case_number}/{len(cases)}] {case}", file=sys.stderr, flush=True)
        command = [
            sys.executable,
            "-m",
            "benchmarks.run",
            "--rows",
            str(rows),
            "--length",
            length,
            "--cardinality",
            str(cardinality),
            "--content",
            content,
            "--input-dtype",
            input_dtype,
            "--tokenizer",
            tokenizer,
            "--null-rate",
            str(args.null_rate),
            "--seed",
            str(args.seed),
            "--warm-repeats",
            str(args.warm_repeats),
        ]
        environment = {**os.environ, "POLARS_MAX_THREADS": str(threads)}
        plugin_report = run_implementation(command, environment, "plugin")
        reference_report = (
            None if args.skip_reference else run_implementation(command, environment, "reference")
        )
        reports.append(combine_reports(plugin_report, reference_report))

    report = {
        "schema_version": 4,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases": reports,
        "skipped": skipped,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")

    flat_rows = [row for case in reports for row in flatten_report(case)]
    if args.csv:
        write_csv(args.csv, flat_rows)
    if args.parquet:
        args.parquet.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(flat_rows).write_parquet(args.parquet)


if __name__ == "__main__":
    main()
