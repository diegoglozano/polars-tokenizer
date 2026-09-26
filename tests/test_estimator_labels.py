"""Exact-count oracle labels are tied to the validated candidate corpus."""

import json
from dataclasses import asdict

import pytest

from benchmarks.estimator_corpus import CorpusRecord, validate_corpus
from benchmarks.estimator_labels import label_records, load_records


def sample_records():
    return [
        CorpusRecord("b", "b", "held_out", "fixture", "en", "prose", ""),
        CorpusRecord("a", "a", "train", "fixture", "en", "prose", "hello world"),
    ]


def test_exact_labels_are_order_independent():
    records = sample_records()
    first, manifest = label_records(records)
    second, reversed_manifest = label_records(list(reversed(records)))
    assert first == second
    assert manifest == reversed_manifest
    assert [label.sample_id for label in first] == ["a", "b"]
    assert first[0].cl100k_base == 2
    assert first[0].o200k_base == 2
    assert first[1].cl100k_base == first[1].o200k_base == 0
    assert manifest["corpus_sha256"] == validate_corpus(records).sha256
    assert manifest["total_tokens"] == {"cl100k_base": 2, "o200k_base": 2}


def test_loader_checks_manifest(tmp_path):
    records = sample_records()
    (tmp_path / "records.jsonl").write_text(
        "".join(json.dumps(asdict(record)) + "\n" for record in records), encoding="utf-8"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(validate_corpus(records))), encoding="utf-8")
    assert load_records(tmp_path) == sorted(records, key=lambda record: record.sample_id)
    manifest_path.write_text(json.dumps({"sha256": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        load_records(tmp_path)
