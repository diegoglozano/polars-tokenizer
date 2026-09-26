"""Generate independently versioned exact-count labels for a candidate corpus.

This benchmark helper is not a raw-text estimator or a product API. It uses
the pinned development-only tiktoken oracle to label the two supported local
tokenizer definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import tiktoken

from benchmarks.estimator_corpus import CorpusRecord, validate_corpus

ORACLE_VERSION = "0.12.0"
ENCODINGS = ("cl100k_base", "o200k_base")


@dataclass(frozen=True, slots=True)
class ExactLabel:
    sample_id: str
    cl100k_base: int
    o200k_base: int


def load_records(directory: Path) -> list[CorpusRecord]:
    """Read JSONL and require its recorded corpus identity to match the data."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    records = [
        CorpusRecord(**json.loads(line))
        for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    actual = validate_corpus(records)
    if manifest != json.loads(json.dumps(asdict(actual))):
        raise ValueError("corpus manifest does not match records")
    return sorted(records, key=lambda record: record.sample_id)


def label_records(records: list[CorpusRecord]) -> tuple[list[ExactLabel], dict]:
    """Count raw text with tiktoken 0.12.0, without special-token rejection."""
    if version("tiktoken") != ORACLE_VERSION:
        raise RuntimeError(f"tiktoken {ORACLE_VERSION} is required")
    corpus = validate_corpus(records)
    encoders = {name: tiktoken.get_encoding(name) for name in ENCODINGS}
    labels = [
        ExactLabel(
            sample_id=record.sample_id,
            **{
                name: len(encoder.encode(record.text, disallowed_special=()))
                for name, encoder in encoders.items()
            },
        )
        for record in sorted(records, key=lambda record: record.sample_id)
    ]
    digest = hashlib.sha256(b"polars-tokenizer-exact-labels-v1")
    for label in labels:
        payload = json.dumps(asdict(label), sort_keys=True, separators=(",", ":")).encode()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    manifest = {
        "schema_version": 1,
        "corpus_sha256": corpus.sha256,
        "oracle": "tiktoken",
        "oracle_version": ORACLE_VERSION,
        "encodings": list(ENCODINGS),
        "records": len(labels),
        "sha256": digest.hexdigest(),
        "total_tokens": {name: sum(getattr(label, name) for label in labels) for name in ENCODINGS},
    }
    return labels, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path, help="directory produced by common_voice_corpus")
    args = parser.parse_args()
    labels, manifest = label_records(load_records(args.corpus_dir))
    (args.corpus_dir / "exact-labels.jsonl").write_text(
        "".join(json.dumps(asdict(label)) + "\n" for label in labels), encoding="utf-8"
    )
    (args.corpus_dir / "exact-labels-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
