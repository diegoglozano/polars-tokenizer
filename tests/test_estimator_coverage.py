"""Coverage summaries keep split, content, and exact-label alignment visible."""

import pytest

from benchmarks.estimator_corpus import CorpusRecord, validate_corpus
from benchmarks.estimator_coverage import summarize_coverage
from benchmarks.estimator_labels import ExactLabel


def fixture():
    records = [
        CorpusRecord("a", "a", "train", "source", "en", "prose", "hello"),
        CorpusRecord("b", "b", "held_out", "source", "en", "prose", ""),
        CorpusRecord("c", "c", "held_out", "other", "und", "python_code", "x = 1\n"),
    ]
    labels = [
        ExactLabel("c", 4, 4),
        ExactLabel("a", 1, 1),
        ExactLabel("b", 0, 0),
    ]
    return records, labels


def test_summarizes_each_content_split():
    records, labels = fixture()
    result = summarize_coverage(records, labels)
    assert result["corpus_sha256"] == validate_corpus(records).sha256
    assert result["records"] == 3
    prose = result["by_content_type"]["prose"]
    assert prose["train"]["utf8_length_classes"]["tiny"] == 1
    assert prose["held_out"]["min_utf8_bytes"] == 0
    assert prose["held_out"]["tokens"]["o200k_base"]["zero_count_records"] == 1
    assert result["by_content_type"]["python_code"]["held_out"]["languages"] == ["und"]
    assert result["by_content_type"]["python_code"]["train"]["records"] == 0
    assert result["by_content_type"]["python_code"]["train"]["tokens"]["o200k_base"]["p95"] is None
    assert result == summarize_coverage(list(reversed(records)), list(reversed(labels)))


def test_rejects_missing_or_duplicate_labels():
    records, labels = fixture()
    with pytest.raises(ValueError, match="one-to-one"):
        summarize_coverage(records, labels[:-1])
    with pytest.raises(ValueError, match="one-to-one"):
        summarize_coverage(records, [*labels, labels[0]])


def test_rejects_invalid_counts():
    records, labels = fixture()
    labels[0] = ExactLabel("c", -1, 4)
    with pytest.raises(ValueError, match="nonnegative"):
        summarize_coverage(records, labels)
