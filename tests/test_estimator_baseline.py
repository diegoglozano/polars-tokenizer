"""The experimental byte model uses train rows and scores held-out rows."""

import math

import pytest

from benchmarks.estimator_baseline import byte_features, evaluate_baseline, fit_byte_model
from benchmarks.estimator_corpus import CorpusRecord
from benchmarks.estimator_labels import ExactLabel


def test_utf8_byte_features():
    assert byte_features("Aé🙂") == (1, 6)
    assert byte_features("") == (0, 0)


def test_fit_two_nonnegative_coefficients():
    model = fit_byte_model([("aa", 2), ("é", 4), ("aaaé", 7)])
    assert math.isclose(model.ascii_tokens_per_byte, 1)
    assert math.isclose(model.nonascii_tokens_per_byte, 2)
    assert math.isclose(model.predict("aé"), 5)


def test_singular_training_set_and_invalid_counts():
    model = fit_byte_model([("a", 1), ("aa", 2)])
    assert model.predict("aaa") == 3
    assert model.nonascii_tokens_per_byte == 0
    with pytest.raises(ValueError, match="training"):
        fit_byte_model([])
    with pytest.raises(ValueError, match="nonnegative"):
        fit_byte_model([("hello", -1)])


def test_held_out_label_does_not_change_fit():
    records = [
        CorpusRecord("a", "a", "train", "source", "en", "prose", "aa"),
        CorpusRecord("b", "b", "train", "source", "fr", "prose", "é"),
        CorpusRecord("c", "c", "held_out", "other", "en", "prose", "aaa"),
    ]
    labels = [ExactLabel("a", 2, 2), ExactLabel("b", 4, 4), ExactLabel("c", 3, 3)]
    first = evaluate_baseline(records, labels, "o200k_base")
    labels[-1] = ExactLabel("c", 300, 300)
    second = evaluate_baseline(records, labels, "o200k_base")
    assert first["fitted_coefficients"] == second["fitted_coefficients"]
    assert first["fitted_ascii_nonascii"]["overall"]["mae_tokens"] == 0
    assert second["fitted_ascii_nonascii"]["overall"]["mae_tokens"] > 0


def test_rejects_label_mismatch():
    records = [
        CorpusRecord("a", "a", "train", "source", "en", "prose", "a"),
        CorpusRecord("b", "b", "held_out", "source", "en", "prose", "b"),
    ]
    with pytest.raises(ValueError, match="one-to-one"):
        evaluate_baseline(records, [ExactLabel("a", 1, 1)], "o200k_base")
