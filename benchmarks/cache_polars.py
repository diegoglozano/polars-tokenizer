"""Run isolated Polars-output cache experiments without changing the plugin."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks.run import machine_metadata

ROOT = Path(__file__).resolve().parents[1]
TOKENIZERS = ("o200k_base", "cl100k_base")
CARDINALITIES = (0.0001, 0.001, 0.01, 0.1, 0.5, 1.0)
MODES = ("uncached", "cache_256", "cache_4096")
ROWS = 100_000


def comma_separated(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one value")
    return parsed


def run_mode(
    tokenizer: str, unique: int, null_rows: int, repeats: int, mode: str
) -> dict[str, Any]:
    environment = {
        **os.environ,
        "POLARS_TOKENIZER_CACHE_POLARS_MODE": mode,
        "POLARS_TOKENIZER_CACHE_POLARS_TOKENIZER": tokenizer,
        "POLARS_TOKENIZER_CACHE_POLARS_UNIQUE": str(unique),
        "POLARS_TOKENIZER_CACHE_POLARS_NULL_ROWS": str(null_rows),
        "POLARS_TOKENIZER_CACHE_POLARS_REPEATS": str(repeats),
    }
    completed = subprocess.run(
        ["cargo", "bench", "--quiet", "--bench", "cache_polars"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"cache_polars {mode} failed:\n{completed.stderr}")
    return json.loads(completed.stdout)


def combine_case(reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = {report["mode"]: report for report in reports}
    if len(reports) != len(MODES) or set(by_mode) != set(MODES):
        raise ValueError("expected one report for each cache mode")
    baseline = by_mode["uncached"]
    identity = ("tokenizer", "rows", "unique_values", "null_rows", "bytes", "dataset_sha256")
    for report in reports:
        if any(report[field] != baseline[field] for field in identity):
            raise ValueError("isolated cache datasets differ")
        if (
            report["counts_sha256"] != baseline["counts_sha256"]
            or report["total_tokens"] != baseline["total_tokens"]
        ):
            raise ValueError("cached output differs from uncached output")

    return {
        "tokenizer": baseline["tokenizer"],
        "rows": baseline["rows"],
        "unique_values": baseline["unique_values"],
        "null_rows": baseline["null_rows"],
        "bytes": baseline["bytes"],
        "dataset_sha256": baseline["dataset_sha256"],
        "counts_sha256": baseline["counts_sha256"],
        "total_tokens": baseline["total_tokens"],
        "modes": by_mode,
        "speedup_vs_uncached": {
            mode: baseline["median_seconds"] / by_mode[mode]["median_seconds"] for mode in MODES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizers", type=comma_separated, default=list(TOKENIZERS))
    parser.add_argument(
        "--cardinalities",
        type=lambda value: [float(item) for item in comma_separated(value)],
        default=list(CARDINALITIES),
    )
    parser.add_argument("--null-rows", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if unknown := sorted(set(args.tokenizers) - set(TOKENIZERS)):
        parser.error(f"unknown tokenizers: {', '.join(unknown)}")
    if any(cardinality not in CARDINALITIES for cardinality in args.cardinalities):
        parser.error("cardinalities must be selected from 0.0001,0.001,0.01,0.1,0.5,1.0")
    if not 0 <= args.null_rows < ROWS:
        parser.error(f"--null-rows must be in [0, {ROWS})")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    cases = []
    for tokenizer in args.tokenizers:
        for cardinality in args.cardinalities:
            non_null_rows = ROWS - args.null_rows
            unique = max(1, min(non_null_rows, round(non_null_rows * cardinality)))
            reports = []
            for mode in MODES:
                print(
                    f"{tokenizer} cardinality={cardinality} mode={mode}",
                    file=sys.stderr,
                    flush=True,
                )
                reports.append(run_mode(tokenizer, unique, args.null_rows, args.repeats, mode))
            cases.append({"requested_cardinality": cardinality, **combine_case(reports)})

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": machine_metadata(None),
        "cases": cases,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
