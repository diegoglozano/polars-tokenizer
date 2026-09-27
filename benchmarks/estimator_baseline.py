"""Train-only two-feature raw-text estimator experiment on a validated corpus.

This benchmark is not a production estimator or an accuracy claim. It fits
nonnegative token-per-byte coefficients using only training records, then
scores held-out records alongside a fixed UTF-8-bytes/4 baseline.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.estimator_corpus import CorpusRecord, read_corpus, validate_corpus
from benchmarks.estimator_eval import Prediction, score_predictions
from benchmarks.estimator_labels import ENCODINGS, ExactLabel, label_records


@dataclass(frozen=True, slots=True)
class ByteModel:
    ascii_tokens_per_byte: float
    nonascii_tokens_per_byte: float

    def predict(self, text: str) -> float:
        ascii_bytes, nonascii_bytes = byte_features(text)
        return (
            self.ascii_tokens_per_byte * ascii_bytes
            + self.nonascii_tokens_per_byte * nonascii_bytes
        )


def byte_features(text: str) -> tuple[int, int]:
    """Count ASCII and non-ASCII UTF-8 bytes in one pass over encoded text."""
    raw = text.encode("utf-8")
    ascii_bytes = sum(byte < 128 for byte in raw)
    return ascii_bytes, len(raw) - ascii_bytes


def fit_byte_model(samples: list[tuple[str, int]]) -> ByteModel:
    """Solve two-variable nonnegative least squares, including boundary cases."""
    if not samples:
        raise ValueError("at least one training sample is required")
    rows = [(byte_features(text), count) for text, count in samples]
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for _, count in rows):
        raise ValueError("counts must be nonnegative integers")
    aa = math.fsum(a * a for (a, _), _ in rows)
    nn = math.fsum(n * n for (_, n), _ in rows)
    an = math.fsum(a * n for (a, n), _ in rows)
    ay = math.fsum(a * y for (a, _), y in rows)
    ny = math.fsum(n * y for (_, n), y in rows)
    candidates = [(0.0, 0.0)]
    if aa:
        candidates.append((max(0.0, ay / aa), 0.0))
    if nn:
        candidates.append((0.0, max(0.0, ny / nn)))
    determinant = aa * nn - an * an
    if determinant > 0:
        ascii_rate = (ay * nn - ny * an) / determinant
        nonascii_rate = (ny * aa - ay * an) / determinant
        if ascii_rate >= 0 and nonascii_rate >= 0:
            candidates.append((ascii_rate, nonascii_rate))

    def squared_error(coefficients: tuple[float, float]) -> float:
        ascii_rate, nonascii_rate = coefficients
        return math.fsum((ascii_rate * a + nonascii_rate * n - y) ** 2 for (a, n), y in rows)

    ascii_rate, nonascii_rate = min(candidates, key=lambda item: (squared_error(item), item))
    return ByteModel(ascii_rate, nonascii_rate)


def evaluate_baseline(records: list[CorpusRecord], labels: list[ExactLabel], encoding: str) -> dict:
    """Fit on train IDs only; compare both raw-text baselines on held-out IDs."""
    corpus = validate_corpus(records)
    if encoding not in ENCODINGS:
        raise ValueError(f"unsupported encoding: {encoding}")
    by_id = {label.sample_id: label for label in labels}
    if len(by_id) != len(labels) or set(by_id) != {record.sample_id for record in records}:
        raise ValueError("exact labels must match corpus sample IDs one-to-one")
    train = [
        (record.text, getattr(by_id[record.sample_id], encoding))
        for record in records
        if record.split == "train"
    ]
    model = fit_byte_model(train)
    held_out = [record for record in records if record.split == "held_out"]

    def predictions(predict) -> list[Prediction]:
        return [
            Prediction(
                text=record.text,
                actual_tokens=getattr(by_id[record.sample_id], encoding),
                estimated_tokens=predict(record.text),
                language=record.language,
                content_type=record.content_type,
            )
            for record in held_out
        ]

    return {
        "schema_version": 1,
        "corpus_sha256": corpus.sha256,
        "encoding": encoding,
        "train_records": corpus.train_records,
        "held_out_records": corpus.held_out_records,
        "fitted_coefficients": asdict(model),
        "fixed_bytes_div_4": score_predictions(
            predictions(lambda text: len(text.encode("utf-8")) / 4)
        ),
        "fitted_ascii_nonascii": score_predictions(predictions(model.predict)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", type=Path)
    args = parser.parse_args()
    records = read_corpus(args.corpus_dir)
    labels, label_manifest = label_records(records)
    reports = {name: evaluate_baseline(records, labels, name) for name in ENCODINGS}
    print(
        json.dumps({"exact_labels_sha256": label_manifest["sha256"], "reports": reports}, indent=2)
    )


if __name__ == "__main__":
    main()
