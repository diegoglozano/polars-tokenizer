"""Compare exact kernel, Polars input iteration, and Arrow output in isolated processes."""

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
MODES = ("kernel_vec", "column_vec", "column_arrow", "output_only")
LENGTH_BYTES = {"tiny": 24, "short": 128, "medium": 2_048}


def run_mode(tokenizer: str, target_bytes: int, repeats: int, mode: str) -> dict[str, Any]:
    environment = {
        **os.environ,
        "POLARS_TOKENIZER_COMPONENT_MODE": mode,
        "POLARS_TOKENIZER_COMPONENT_TOKENIZER": tokenizer,
        "POLARS_TOKENIZER_COMPONENT_TARGET_BYTES": str(target_bytes),
        "POLARS_TOKENIZER_COMPONENT_REPEATS": str(repeats),
    }
    completed = subprocess.run(
        ["cargo", "bench", "--quiet", "--bench", "component_overhead"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"component_overhead {mode} failed:\n{completed.stderr}")
    return json.loads(completed.stdout)


def combine_case(reports: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode = {report["mode"]: report for report in reports}
    if len(reports) != len(MODES) or set(by_mode) != set(MODES):
        raise ValueError("expected one report for each component mode")
    baseline = by_mode["kernel_vec"]
    identity = (
        "tokenizer",
        "rows",
        "target_bytes",
        "null_rows",
        "unique_values",
        "bytes",
        "dataset_sha256",
    )
    for report in reports:
        if any(report[field] != baseline[field] for field in identity):
            raise ValueError("isolated component datasets differ")
        if (
            report["counts_sha256"] != baseline["counts_sha256"]
            or report["total_tokens"] != baseline["total_tokens"]
        ):
            raise ValueError("component output differs from kernel output")

    return {
        **{field: baseline[field] for field in (*identity, "counts_sha256", "total_tokens")},
        "modes": by_mode,
        "time_ratio_vs_kernel_vec": {
            mode: by_mode[mode]["median_seconds"] / baseline["median_seconds"] for mode in MODES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizers", nargs="+", default=list(TOKENIZERS), choices=TOKENIZERS)
    parser.add_argument(
        "--lengths", nargs="+", default=list(LENGTH_BYTES), choices=tuple(LENGTH_BYTES)
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    cases = []
    for tokenizer in args.tokenizers:
        for length in args.lengths:
            reports = []
            for mode in MODES:
                print(f"{tokenizer} length={length} mode={mode}", file=sys.stderr, flush=True)
                reports.append(run_mode(tokenizer, LENGTH_BYTES[length], args.repeats, mode))
            cases.append({"length": length, **combine_case(reports)})
    result = {
        "schema_version": 2,
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
