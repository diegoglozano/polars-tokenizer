"""Build a source-separated Python-code candidate from two pinned MIT projects.

Flask source files are training records and Black source files are held out.
This avoids splitting related files inside one project, but is not a broad
sample of programming languages or code repositories.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarks.estimator_corpus import CorpusRecord, Split, read_corpus, validate_corpus


@dataclass(frozen=True, slots=True)
class CodeSource:
    repository: str
    revision: str
    archive_sha256: str
    package_prefix: str
    file_count: int
    split: Split


SOURCES = (
    CodeSource(
        "pallets/flask",
        "d73fa1cdcbd8b1465c151db8924ba58b1dd14e35",
        "f2859ca2addf5a0c09e66f79a97851c26ea534727970596fd9e1f643e469a827",
        "src/flask/",
        24,
        "train",
    ),
    CodeSource(
        "psf/black",
        "8d5a2d9f49378d7abe2eb632df1601818de8c24e",
        "160cc38cb9694cb95bf1d4fd40c859cf7ae30e6e311a564a3ffdb05e975466b2",
        "src/black/",
        25,
        "held_out",
    ),
)


def download_archives(sources: tuple[CodeSource, ...] = SOURCES) -> dict[str, bytes]:
    archives = {}
    for source in sources:
        url = f"https://github.com/{source.repository}/archive/{source.revision}.tar.gz"
        with urllib.request.urlopen(url, timeout=30) as response:
            archives[source.repository] = response.read(5_000_001)
    return archives


def build_records(
    archives: Mapping[str, bytes], sources: tuple[CodeSource, ...] = SOURCES
) -> list[CorpusRecord]:
    """Verify both archives and retain each whole Python source file."""
    if set(archives) != {source.repository for source in sources}:
        raise ValueError("input archives do not match pinned code sources")
    records = []
    for source in sources:
        raw = archives[source.repository]
        if hashlib.sha256(raw).hexdigest() != source.archive_sha256:
            raise ValueError(f"code archive checksum mismatch: {source.repository}")
        repo_name = source.repository.split("/")[-1]
        prefix = f"{repo_name}-{source.revision}/{source.package_prefix}"
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            members = sorted(
                (
                    member
                    for member in archive.getmembers()
                    if member.name.startswith(prefix)
                    and member.name.endswith(".py")
                    and member.isfile()
                ),
                key=lambda member: member.name,
            )
            if len(members) != source.file_count or len({member.name for member in members}) != len(
                members
            ):
                raise ValueError(f"unexpected Python file set: {source.repository}")
            for member in members:
                file = archive.extractfile(member)
                if file is None:
                    raise ValueError(f"missing archive member: {member.name}")
                text = file.read().decode("utf-8")
                relative_path = member.name[len(f"{repo_name}-{source.revision}/") :]
                identity = f"{source.repository}@{source.revision}:{relative_path}"
                records.append(
                    CorpusRecord(
                        sample_id=identity,
                        family_id=identity,
                        split=source.split,
                        source_id=identity,
                        language="und",
                        content_type="python_code",
                        text=text,
                    )
                )
    validate_corpus(records)
    return sorted(records, key=lambda record: record.sample_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--base-corpus", type=Path, help="validated corpus directory to combine")
    args = parser.parse_args()
    records = build_records(download_archives())
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
