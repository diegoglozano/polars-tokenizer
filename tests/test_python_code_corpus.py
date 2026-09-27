"""Offline checks for source-separated Python-code corpus imports."""

import hashlib
import io
import tarfile

import pytest

from benchmarks.estimator_corpus import validate_corpus
from benchmarks.python_code_corpus import CodeSource, build_records


def archive(repository: str, revision: str, texts: list[str]) -> bytes:
    stream = io.BytesIO()
    repo_name = repository.split("/")[-1]
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for index, text in enumerate(texts):
            data = text.encode()
            member = tarfile.TarInfo(f"{repo_name}-{revision}/src/{repo_name}/file{index}.py")
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return stream.getvalue()


def fixture():
    files = {
        "owner/alpha": archive("owner/alpha", "rev1", ["a = 1\n", "b = 2\n"]),
        "owner/beta": archive("owner/beta", "rev2", ["c = 3\n", "d = 4\n"]),
    }
    sources = (
        CodeSource(
            "owner/alpha",
            "rev1",
            hashlib.sha256(files["owner/alpha"]).hexdigest(),
            "src/alpha/",
            2,
            "train",
        ),
        CodeSource(
            "owner/beta",
            "rev2",
            hashlib.sha256(files["owner/beta"]).hexdigest(),
            "src/beta/",
            2,
            "held_out",
        ),
    )
    return files, sources


def test_imports_source_separated_files():
    files, sources = fixture()
    records = build_records(files, sources)
    assert len(records) == 4
    assert all(record.content_type == "python_code" for record in records)
    assert all(record.language == "und" for record in records)
    assert {record.split for record in records if "alpha" in record.source_id} == {"train"}
    assert {record.split for record in records if "beta" in record.source_id} == {"held_out"}
    assert validate_corpus(records).records == 4
    assert records == build_records(files, sources)


def test_rejects_archive_checksum_change():
    files, sources = fixture()
    files["owner/alpha"] += b"tampered"
    with pytest.raises(ValueError, match="checksum"):
        build_records(files, sources)


def test_rejects_cross_source_duplicate_text():
    files, sources = fixture()
    files["owner/beta"] = archive("owner/beta", "rev2", ["a = 1\n", "d = 4\n"])
    sources = (
        sources[0],
        CodeSource(
            "owner/beta",
            "rev2",
            hashlib.sha256(files["owner/beta"]).hexdigest(),
            "src/beta/",
            2,
            "held_out",
        ),
    )
    with pytest.raises(ValueError, match="identical text"):
        build_records(files, sources)
