"""Reproduce a small, pinned Common Voice sentence corpus for estimator work.

Only sentence-collector.txt files are used; other Common Voice text files may
have different provenance. This is a prose-only candidate, not a representative
multi-format evaluation corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.estimator_corpus import CorpusRecord, validate_corpus

REVISION = "3ae618d5b34381ab154bcac562ed81aacf93e42c"
BASE_URL = f"https://raw.githubusercontent.com/common-voice/common-voice/{REVISION}/server/data"


@dataclass(frozen=True, slots=True)
class SourceFile:
    language: str
    sha256: str


SOURCES = (
    SourceFile("en", "31ac8e200449ebd7aee4a8c6bd16d4dafd076cf11f7c3efe1035f0283712d136"),
    SourceFile("es", "0a125f0d1cd892c5d727abb8006f5fb964ef9fe1e41bdf4afd3c4b845c47286b"),
    SourceFile("fr", "214f3d5b0110e86e3f2018ca6971b918db8ad5cc4c5612ebab58cd0849fb0715"),
    SourceFile("de", "de6bd33fc5742350690fa4be6197aafd9be2c6c9fe4b70b8c5c8670971acc117"),
    SourceFile("ja", "964acfaa4c6ed0f968fbeb152300ab800bb30fce5b2e4c234201ffb3a971598b"),
    SourceFile("zh-CN", "c3431534ec4dc0a087546c6e9f254295babec6a42f70bce738da629d41c76ca8"),
    SourceFile("ar", "438817d8a0de050edfe4d1da10c7b3d9d77702db44d446f9e08e53f012f6e27e"),
    SourceFile("hi", "0de9bc2283188445d3a564974a688083fe6d9c29f11c3c31b4e01dc25af1885c"),
    SourceFile("ru", "e0869a008310c452f63fd02e4d52f8abc58df2de73e558c486719161e3ad1b87"),
    SourceFile("sw", "ceaa6ec4d974f375a70c5f7a5e709f88a08a4f31f964b11a5a43b96e1a3d4db9"),
)


def download_sources(sources: tuple[SourceFile, ...] = SOURCES) -> dict[str, bytes]:
    """Download only pinned sentence-collector files; caller verifies bytes."""
    result = {}
    for source in sources:
        url = f"{BASE_URL}/{source.language}/sentence-collector.txt"
        with urllib.request.urlopen(url, timeout=30) as response:
            result[source.language] = response.read()
    return result


def build_records(
    files: Mapping[str, bytes],
    sources: tuple[SourceFile, ...] = SOURCES,
    *,
    train_per_language: int = 256,
    held_out_per_language: int = 64,
) -> list[CorpusRecord]:
    """Select hash-stable 80/20 text-family splits, bounded per language."""
    if train_per_language < 1 or held_out_per_language < 1:
        raise ValueError("both per-language limits must be positive")
    if set(files) != {source.language for source in sources}:
        raise ValueError("input languages do not match pinned sources")

    records = []
    for source in sources:
        raw = files[source.language]
        if hashlib.sha256(raw).hexdigest() != source.sha256:
            raise ValueError(f"source checksum mismatch: {source.language}")
        text = raw.decode("utf-8")
        candidates: dict[str, list[tuple[str, str, int]]] = {"train": [], "held_out": []}
        seen: set[str] = set()
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line or line in seen:
                continue
            seen.add(line)
            family = hashlib.sha256(line.encode("utf-8")).hexdigest()
            split = "held_out" if int(family[:8], 16) % 5 == 0 else "train"
            candidates[split].append((family, line, line_number))
        source_id = f"common-voice@{REVISION}:server/data/{source.language}/sentence-collector.txt"
        for split, limit in (("train", train_per_language), ("held_out", held_out_per_language)):
            selected = sorted(candidates[split])[:limit]
            if len(selected) != limit:
                raise ValueError(f"insufficient {split} sentences for {source.language}")
            records.extend(
                CorpusRecord(
                    sample_id=f"{source.language}:{line_number}",
                    family_id=family,
                    split=split,
                    source_id=source_id,
                    language=source.language,
                    content_type="prose",
                    text=line,
                )
                for family, line, line_number in selected
            )
    validate_corpus(records)
    return sorted(records, key=lambda record: record.sample_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="directory for JSONL records and manifest")
    args = parser.parse_args()
    records = build_records(download_sources())
    manifest = validate_corpus(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(asdict(manifest), indent=2))


if __name__ == "__main__":
    main()
