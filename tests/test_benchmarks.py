from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks._data import CONTENT_TYPES, dataset_digest, dataset_statistics, make_dataset
from benchmarks.gemini_local import (
    google_batch_total,
    google_scalar,
    sentencepiece_batch_ids,
    sentencepiece_scalar_ids,
)
from benchmarks.gemini_local import rate as gemini_rate
from benchmarks.matrix import flatten_report


def test_dataset_is_deterministic_and_guarantees_cardinality() -> None:
    first = make_dataset(100, "short", 0.1, 42, content="code")
    second = make_dataset(100, "short", 0.1, 42, content="code")

    assert first == second
    assert len(set(first)) == 10
    assert dataset_digest(first) == dataset_digest(second)


@pytest.mark.parametrize("content", CONTENT_TYPES)
def test_every_content_type_generates_valid_dataset(content: str) -> None:
    values = make_dataset(10, "tiny", 1.0, 7, content=content)
    assert len(values) == 10
    assert all(isinstance(value, str) for value in values)


def test_null_statistics_and_digest_are_stable() -> None:
    values = make_dataset(100, "tiny", 0.2, 9, null_rate=0.1)
    statistics = dataset_statistics(values)

    assert statistics["rows"] == 100
    assert statistics["null_rows"] == 10
    assert statistics["unique_values"] == 18
    assert statistics["actual_cardinality"] == 0.2
    assert statistics["bytes"] > 0
    assert dataset_digest(values) == dataset_digest(list(values))


def test_known_unique_count_avoids_recomputing_cardinality() -> None:
    values = make_dataset(10, "tiny", 1.0, 3)
    statistics = dataset_statistics(values, unique_values=10)
    assert statistics["unique_values"] == 10


def test_all_null_statistics_are_defined() -> None:
    statistics = dataset_statistics([None, None])
    assert statistics["null_rows"] == 2
    assert statistics["actual_cardinality"] == 0.0


def test_invalid_dataset_parameters_fail() -> None:
    with pytest.raises(ValueError, match="rows"):
        make_dataset(0, "tiny", 1.0, 1)
    with pytest.raises(ValueError, match="cardinality"):
        make_dataset(1, "tiny", 0.0, 1)
    with pytest.raises(ValueError, match="null_rate"):
        make_dataset(1, "tiny", 1.0, 1, null_rate=1.0)


def test_flatten_report_keeps_comparison_fields() -> None:
    measurement = {
        "seconds": 0.5,
        "rows_per_second": 20.0,
        "mib_per_second": 2.0,
        "tokens_per_second": 30.0,
        "process_cpu_percent": 150.0,
    }
    report = {
        "dataset": {
            "rows": 10,
            "bytes": 100,
            "length_class": "tiny",
            "content": "mixed",
            "input_dtype": "categorical",
            "tokenizer": "cl100k_base",
            "requested_cardinality": 1.0,
            "actual_cardinality": 1.0,
            "null_rate": 0.0,
            "sha256": "abc",
        },
        "environment": {
            "polars_threads": 2,
            "git_commit": "deadbeef",
            "git_dirty": True,
        },
        "measurements": {
            "plugin_cold": measurement,
            "plugin_warm": measurement,
            "python_tiktoken_scalar": None,
            "peak_rss_bytes": 1_024,
        },
    }

    rows = flatten_report(report)
    assert [row["implementation"] for row in rows] == ["plugin_cold", "plugin_warm"]
    assert rows[0]["threads"] == 2
    assert rows[0]["input_dtype"] == "categorical"
    assert rows[0]["tokenizer"] == "cl100k_base"
    assert rows[0]["git_dirty"] is True


def test_gemini_benchmark_rate_uses_median_wall_time() -> None:
    samples = [
        {"wall_seconds": 2.0, "cpu_seconds": 1.0, "process_cpu_percent": 50.0},
        {"wall_seconds": 1.0, "cpu_seconds": 1.0, "process_cpu_percent": 100.0},
        {"wall_seconds": 3.0, "cpu_seconds": 1.0, "process_cpu_percent": 33.0},
    ]

    measurement = gemini_rate(samples, rows=20, byte_count=2 * 1024 * 1024, token_count=30)

    assert measurement["seconds"] == 2.0
    assert measurement["best_seconds"] == 1.0
    assert measurement["rows_per_second"] == 10.0
    assert measurement["mib_per_second"] == 1.0
    assert measurement["tokens_per_second"] == 15.0


def test_gemini_benchmark_paths_preserve_counts() -> None:
    class FakeTokenizer:
        def count_tokens(self, contents: str | list[str]) -> SimpleNamespace:
            texts = [contents] if isinstance(contents, str) else contents
            return SimpleNamespace(total_tokens=sum(len(text) for text in texts))

    class FakeProcessor:
        def encode(self, contents: str | list[str]) -> list[int] | list[list[int]]:
            if isinstance(contents, str):
                return list(range(len(contents)))
            return [list(range(len(text))) for text in contents]

    texts = ["a", "three", ""]
    expected_counts = [1, 5, 0]

    google_per_row = google_scalar(FakeTokenizer(), texts)
    google_total = google_batch_total(FakeTokenizer(), texts)
    sentencepiece_per_row = sentencepiece_scalar_ids(FakeProcessor(), texts)
    sentencepiece_batch = sentencepiece_batch_ids(FakeProcessor(), texts)

    assert google_per_row.counts == expected_counts
    assert sentencepiece_per_row.counts == expected_counts
    assert sentencepiece_batch.counts == expected_counts
    assert {
        result.total
        for result in (
            google_per_row,
            google_total,
            sentencepiece_per_row,
            sentencepiece_batch,
        )
    } == {6}
