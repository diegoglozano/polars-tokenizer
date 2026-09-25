from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.estimator_corpus import CorpusRecord, validate_corpus


def sample_records() -> list[CorpusRecord]:
    return [
        CorpusRecord("en-1", "en-source-1", "train", "self-authored", "en", "prose", "Hello."),
        CorpusRecord("en-2", "en-source-1", "train", "self-authored", "en", "prose", "Hi!"),
        CorpusRecord(
            "ja-1", "ja-source-1", "held_out", "self-authored", "ja", "prose", "こんにちは"
        ),
        CorpusRecord("empty-1", "empty-source", "held_out", "self-authored", "und", "empty", ""),
    ]


def test_corpus_manifest_is_order_independent_and_covers_all_fields() -> None:
    records = sample_records()
    manifest = validate_corpus(records)

    assert manifest == validate_corpus(list(reversed(records)))
    assert manifest.records == 4
    assert manifest.train_records == 2
    assert manifest.held_out_records == 2
    assert manifest.languages == ("en", "ja", "und")
    assert manifest.content_types == ("empty", "prose")
    assert manifest.source_ids == ("self-authored",)
    assert manifest.sha256 == "63222101dfddb6f55087911be7eef1feedc3f7de0bef2694db663dcc2e6c6c6b"

    modified = [*records[:-1], replace(records[-1], content_type="other")]
    assert validate_corpus(modified).sha256 != manifest.sha256


def test_duplicate_ids_are_rejected() -> None:
    records = sample_records()
    records.append(replace(records[-1], text="different"))
    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_corpus(records)


def test_related_family_cannot_cross_train_and_held_out() -> None:
    records = sample_records()
    records[-1] = replace(records[-1], family_id="en-source-1")
    with pytest.raises(ValueError, match="family_id crosses"):
        validate_corpus(records)


def test_identical_text_cannot_cross_train_and_held_out() -> None:
    records = sample_records()
    records[-1] = replace(records[-1], text="Hello.")
    with pytest.raises(ValueError, match="identical text crosses"):
        validate_corpus(records)


@pytest.mark.parametrize("split", ["train", "held_out"])
def test_both_partitions_are_required(split: str) -> None:
    records = [replace(record, split=split) for record in sample_records()]  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="both train and held_out"):
        validate_corpus(records)


@pytest.mark.parametrize(
    "field", ["sample_id", "family_id", "source_id", "language", "content_type"]
)
def test_required_labels_are_nonempty(field: str) -> None:
    records = sample_records()
    records[0] = replace(records[0], **{field: ""})
    with pytest.raises(ValueError, match=field):
        validate_corpus(records)


def test_unknown_split_is_rejected() -> None:
    records = sample_records()
    records[0] = replace(records[0], split="test")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="split must be"):
        validate_corpus(records)


def test_empty_corpus_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_corpus([])
