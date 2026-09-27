"""Build a pinned JSON conformance-fixture corpus, optionally with Common Voice.

Each upstream file remains one record and one split family. The fixtures are
useful structured-text coverage, not a representative sample of production JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from dataclasses import asdict
from pathlib import Path

from benchmarks.estimator_corpus import CorpusRecord, read_corpus, validate_corpus

REVISION = "5b0ee1613e45fcc2bddac00e07c19cd49b00d8a8"
ARCHIVE_SHA256 = "baf7510ec75fb87311b8272cd04bab51459449f7c5f12befdc55a41ec19de8bf"
ARCHIVE_URL = f"https://github.com/json-schema-org/JSON-Schema-Test-Suite/archive/{REVISION}.tar.gz"
SOURCE_PREFIX = f"JSON-Schema-Test-Suite-{REVISION}/tests/draft2020-12/"
EXPECTED_FILES = 46


def download_archive() -> bytes:
    with urllib.request.urlopen(ARCHIVE_URL, timeout=30) as response:
        return response.read(2_000_001)


def build_records(
    archive_bytes: bytes, *, expected_sha256: str = ARCHIVE_SHA256
) -> list[CorpusRecord]:
    """Verify the archive and select top-level draft-2020-12 JSON test files."""
    if hashlib.sha256(archive_bytes).hexdigest() != expected_sha256:
        raise ValueError("JSON Schema archive checksum mismatch")
    records = []
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = sorted(
            (
                member
                for member in archive.getmembers()
                if member.name.startswith(SOURCE_PREFIX)
                and member.name[len(SOURCE_PREFIX) :].endswith(".json")
                and "/" not in member.name[len(SOURCE_PREFIX) :]
                and member.isfile()
            ),
            key=lambda member: member.name,
        )
        if len(members) != EXPECTED_FILES or len({member.name for member in members}) != len(
            members
        ):
            raise ValueError("unexpected JSON Schema file set")
        for member in members:
            file = archive.extractfile(member)
            if file is None:
                raise ValueError(f"missing archive member: {member.name}")
            raw = file.read()
            text = raw.decode("utf-8")
            json.loads(text)
            name = member.name[len(SOURCE_PREFIX) :]
            family_id = f"json-schema:draft2020-12:{name}"
            split_hash = hashlib.sha256(family_id.encode()).digest()
            records.append(
                CorpusRecord(
                    sample_id=family_id,
                    family_id=family_id,
                    split="held_out" if split_hash[0] % 5 == 0 else "train",
                    source_id=(f"json-schema-test-suite@{REVISION}:tests/draft2020-12/{name}"),
                    language="en",
                    content_type="json",
                    text=text,
                )
            )
    validate_corpus(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--base-corpus",
        type=Path,
        help="validated corpus directory to combine with this JSON candidate",
    )
    args = parser.parse_args()
    records = build_records(download_archive())
    if args.base_corpus is not None:
        records.extend(read_corpus(args.base_corpus))
    records.sort(key=lambda record: record.sample_id)
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
