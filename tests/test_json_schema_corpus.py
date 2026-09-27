"""Offline checks for the pinned JSON conformance-fixture importer."""

import hashlib
import io
import json
import tarfile

import pytest

from benchmarks.estimator_corpus import validate_corpus
from benchmarks.json_schema_corpus import EXPECTED_FILES, SOURCE_PREFIX, build_records


def fixture_archive(count=EXPECTED_FILES):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for index in range(count):
            name = f"{SOURCE_PREFIX}fixture-{index:02}.json"
            data = json.dumps({"schema": {"type": "integer"}, "index": index}).encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return stream.getvalue()


def test_imports_complete_file_families():
    archive = fixture_archive()
    checksum = hashlib.sha256(archive).hexdigest()
    records = build_records(archive, expected_sha256=checksum)
    assert len(records) == EXPECTED_FILES
    assert {record.content_type for record in records} == {"json"}
    assert {record.language for record in records} == {"en"}
    assert all(record.sample_id == record.family_id for record in records)
    assert all(json.loads(record.text) for record in records)
    assert validate_corpus(records).records == EXPECTED_FILES
    assert records == build_records(archive, expected_sha256=checksum)


def test_rejects_wrong_archive_checksum():
    with pytest.raises(ValueError, match="checksum"):
        build_records(fixture_archive(), expected_sha256="bad")


def test_rejects_missing_file():
    archive = fixture_archive(EXPECTED_FILES - 1)
    with pytest.raises(ValueError, match="file set"):
        build_records(archive, expected_sha256=hashlib.sha256(archive).hexdigest())
