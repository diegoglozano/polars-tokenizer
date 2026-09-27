"""Audit split and format coverage of an exact-labeled estimator corpus.

Coverage describes the candidate's shape; it is not an estimator accuracy
measurement and does not establish representativeness.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.estimator_corpus import CorpusRecord, read_corpus, validate_corpus
from benchmarks.estimator_eval import utf8_length_class
from benchmarks.estimator_labels import ENCODINGS, ExactLabel, label_records


def _nearest_rank(values: list[int], percentile: float) -> int:
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def _group_summary(rows: list[tuple[CorpusRecord, ExactLabel]]) -> dict:
    lengths = [len(record.text.encode("utf-8")) for record, _ in rows]
    buckets = {name: 0 for name in ("tiny", "short", "medium", "long")}
    for record, _ in rows:
        buckets[utf8_length_class(record.text)] += 1
    return {
        "records": len(rows),
        "languages": sorted({record.language for record, _ in rows}),
        "ascii_only_records": sum(record.text.isascii() for record, _ in rows),
        "utf8_length_classes": buckets,
        "min_utf8_bytes": min(lengths),
        "median_utf8_bytes": statistics.median(lengths),
        "max_utf8_bytes": max(lengths),
        "tokens": {
            name: {
                "zero_count_records": sum(getattr(label, name) == 0 for _, label in rows),
                "median": statistics.median(getattr(label, name) for _, label in rows),
                "p95": _nearest_rank([getattr(label, name) for _, label in rows], 0.95),
                "max": max(getattr(label, name) for _, label in rows),
            }
            for name in ENCODINGS
        },
    }


def summarize_coverage(records: list[CorpusRecord], labels: list[ExactLabel]) -> dict:
    """Require an exact one-to-one label join and report each format/split."""
    manifest = validate_corpus(records)
    by_id = {label.sample_id: label for label in labels}
    if len(by_id) != len(labels) or set(by_id) != {record.sample_id for record in records}:
        raise ValueError("exact labels must match corpus sample IDs one-to-one")
    if any(
        isinstance(getattr(label, name), bool)
        or not isinstance(getattr(label, name), int)
        or getattr(label, name) < 0
        for label in labels
        for name in ENCODINGS
    ):
        raise ValueError("exact labels must be nonnegative integers")
    groups: dict[str, dict[str, list[tuple[CorpusRecord, ExactLabel]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        groups[record.content_type][record.split].append((record, by_id[record.sample_id]))
    return {
        "schema_version": 1,
        "corpus_sha256": manifest.sha256,
        "records": manifest.records,
        "by_content_type": {
            content_type: {
                split: _group_summary(group) for split, group in sorted(split_groups.items())
            }
            for content_type, split_groups in sorted(groups.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path)
    args = parser.parse_args()
    records = read_corpus(args.corpus_dir)
    labels, label_manifest = label_records(records)
    result = summarize_coverage(records, labels)
    result["exact_labels_sha256"] = label_manifest["sha256"]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
