from __future__ import annotations

import math

import pytest

from benchmarks.estimator_eval import (
    Prediction,
    score_predictions,
    token_count_class,
    utf8_length_class,
)


def test_error_summary_handles_zero_references_and_strata() -> None:
    report = score_predictions(
        [
            Prediction("hello", 2, 3.0, "en", "prose"),
            Prediction("hola 🌍", 4, 2.0, "es", "prose"),
            Prediction("", 0, 1.0, "und", "empty"),
        ]
    )

    assert report["schema_version"] == 1
    assert report["overall"] == {
        "samples": 3,
        "zero_actual_samples": 1,
        "mae_tokens": pytest.approx(4 / 3),
        "median_absolute_error_tokens": 1.0,
        "bias_tokens": 0.0,
        "mape_percent": 50.0,
        "p50_relative_error_percent": 50.0,
        "p95_relative_error_percent": 50.0,
        "p99_relative_error_percent": 50.0,
        "max_relative_error_percent": 50.0,
    }
    assert report["by"]["ascii_class"]["ascii"]["samples"] == 2
    assert report["by"]["ascii_class"]["non_ascii"]["samples"] == 1
    assert report["by"]["language"]["es"]["bias_tokens"] == -2.0
    assert report["by"]["content_type"]["empty"]["mape_percent"] is None
    assert report["by"]["token_count_class"]["0"]["zero_actual_samples"] == 1


def test_nearest_rank_relative_percentiles_are_explicit() -> None:
    rows = [Prediction("text", 100, float(100 + error), "en", "prose") for error in range(1, 101)]
    overall = score_predictions(rows)["overall"]

    assert overall["p50_relative_error_percent"] == 50.0
    assert overall["p95_relative_error_percent"] == 95.0
    assert overall["p99_relative_error_percent"] == 99.0
    assert overall["max_relative_error_percent"] == 100.0
    assert overall["mape_percent"] == pytest.approx(50.5)


def test_all_zero_references_have_defined_absolute_metrics() -> None:
    overall = score_predictions([Prediction("", 0, 2.0, "und", "empty")])["overall"]

    assert overall["mae_tokens"] == 2.0
    assert overall["bias_tokens"] == 2.0
    assert overall["zero_actual_samples"] == 1
    assert overall["mape_percent"] is None
    assert overall["p50_relative_error_percent"] is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("x" * 24, "tiny"),
        ("x" * 25, "short"),
        ("x" * 128, "short"),
        ("x" * 129, "medium"),
        ("x" * 2_048, "medium"),
        ("x" * 2_049, "long"),
        ("🌍" * 7, "short"),
    ],
)
def test_utf8_length_class_boundaries(text: str, expected: str) -> None:
    assert utf8_length_class(text) == expected


@pytest.mark.parametrize(
    "count,expected",
    [
        (0, "0"),
        (1, "1-8"),
        (8, "1-8"),
        (9, "9-32"),
        (32, "9-32"),
        (33, "33-128"),
        (128, "33-128"),
        (129, "129+"),
    ],
)
def test_token_count_class_boundaries(count: int, expected: str) -> None:
    assert token_count_class(count) == expected


@pytest.mark.parametrize("count", [-1, True])
def test_invalid_token_count_class_input_fails(count: int) -> None:
    with pytest.raises(ValueError, match="actual_tokens"):
        token_count_class(count)


@pytest.mark.parametrize(
    "row,match",
    [
        (Prediction("text", -1, 1.0, "en", "prose"), "actual_tokens"),
        (Prediction("text", True, 1.0, "en", "prose"), "actual_tokens"),
        (Prediction("text", 1, -1.0, "en", "prose"), "estimated_tokens"),
        (Prediction("text", 1, math.nan, "en", "prose"), "estimated_tokens"),
        (Prediction("text", 1, math.inf, "en", "prose"), "estimated_tokens"),
        (Prediction("text", 1, True, "en", "prose"), "estimated_tokens"),
        (Prediction("text", 1, 1.0, "", "prose"), "labels"),
    ],
)
def test_invalid_predictions_fail(row: Prediction, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        score_predictions([row])


def test_empty_predictions_fail() -> None:
    with pytest.raises(ValueError, match="at least one"):
        score_predictions([])
