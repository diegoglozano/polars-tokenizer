"""Identity and leakage checks for a future raw-text estimator corpus.

This module validates supplied records. It does not provide a corpus or
decide whether a dataset is representative or legally distributable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, TypeAlias

Split: TypeAlias = Literal["train", "held_out"]
FIELDS = ("sample_id", "family_id", "split", "source_id", "language", "content_type", "text")


@dataclass(frozen=True, slots=True)
class CorpusRecord:
    """One text; related variants share a family and must share a split."""

    sample_id: str
    family_id: str
    split: Split
    source_id: str
    language: str
    content_type: str
    text: str


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """Order-independent identity and basic coverage of validated records."""

    sha256: str
    records: int
    train_records: int
    held_out_records: int
    languages: tuple[str, ...]
    content_types: tuple[str, ...]
    source_ids: tuple[str, ...]


def _hash_field(digest: hashlib._Hash, value: str) -> None:
    encoded = value.encode()
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def validate_corpus(records: list[CorpusRecord]) -> CorpusManifest:
    """Reject duplicate IDs, cross-split families/text, and missing partitions.

    The SHA-256 covers every record field, sorted by sample ID, with byte-length
    prefixes and a schema marker. An empty text is valid for zero-token tests.
    """
    if not records:
        raise ValueError("corpus must contain at least one record")

    by_id: dict[str, CorpusRecord] = {}
    family_splits: dict[str, Split] = {}
    text_splits: dict[str, Split] = {}
    for record in records:
        for field in FIELDS:
            value = getattr(record, field)
            if not isinstance(value, str) or (field != "text" and not value):
                raise ValueError(f"{field} must be a nonempty string")
        if record.split not in ("train", "held_out"):
            raise ValueError("split must be train or held_out")
        if record.sample_id in by_id:
            raise ValueError(f"duplicate sample_id: {record.sample_id!r}")
        by_id[record.sample_id] = record

        previous_family_split = family_splits.setdefault(record.family_id, record.split)
        if previous_family_split != record.split:
            raise ValueError(f"family_id crosses train and held_out: {record.family_id!r}")
        previous_text_split = text_splits.setdefault(record.text, record.split)
        if previous_text_split != record.split:
            raise ValueError("identical text crosses train and held_out")

    train_records = sum(record.split == "train" for record in records)
    held_out_records = len(records) - train_records
    if not train_records or not held_out_records:
        raise ValueError("corpus must contain both train and held_out records")

    digest = hashlib.sha256(b"polars-tokenizer-estimator-corpus-v1")
    for sample_id in sorted(by_id):
        record = by_id[sample_id]
        for field in FIELDS:
            _hash_field(digest, getattr(record, field))
    return CorpusManifest(
        sha256=digest.hexdigest(),
        records=len(records),
        train_records=train_records,
        held_out_records=held_out_records,
        languages=tuple(sorted({record.language for record in records})),
        content_types=tuple(sorted({record.content_type for record in records})),
        source_ids=tuple(sorted({record.source_id for record in records})),
    )
