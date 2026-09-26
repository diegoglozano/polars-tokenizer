"""Offline tests for the pinned external-corpus sampler."""

import hashlib
import unittest

from benchmarks.common_voice_corpus import SourceFile, build_records
from benchmarks.estimator_corpus import validate_corpus


def fixture():
    files = {
        "en": b"shared\nEnglish one\nEnglish two\nEnglish three\nEnglish four\n",
        "es": "shared\nEspañol uno\nEspañol dos\nEspañol tres\nEspañol cuatro\n".encode(),
    }
    sources = tuple(
        SourceFile(lang, hashlib.sha256(raw).hexdigest()) for lang, raw in files.items()
    )
    return files, sources


class CommonVoiceCorpusTests(unittest.TestCase):
    def test_reproducible_with_no_cross_split_text(self):
        files, sources = fixture()
        # The small fixture has no guaranteed hash split balance; use a larger
        # deterministic set to exercise successful selection in both languages.
        files = {
            lang: raw + b"".join(f"{lang} sample {i}\n".encode() for i in range(100))
            for lang, raw in files.items()
        }
        sources = tuple(
            SourceFile(lang, hashlib.sha256(raw).hexdigest()) for lang, raw in files.items()
        )
        first = build_records(files, sources, train_per_language=3, held_out_per_language=2)
        second = build_records(files, sources, train_per_language=3, held_out_per_language=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(validate_corpus(first).records, 10)
        self.assertEqual({record.language for record in first}, {"en", "es"})

    def test_rejects_checksum_change(self):
        files, sources = fixture()
        files["en"] += b"changed\n"
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            build_records(files, sources, train_per_language=1, held_out_per_language=1)

    def test_rejects_missing_source(self):
        files, sources = fixture()
        with self.assertRaisesRegex(ValueError, "input languages"):
            build_records({"en": files["en"]}, sources)

    def test_rejects_insufficient_split(self):
        files, sources = fixture()
        with self.assertRaisesRegex(ValueError, "insufficient"):
            build_records(files, sources, train_per_language=100, held_out_per_language=100)

    def test_rejects_nonpositive_limit(self):
        files, sources = fixture()
        with self.assertRaisesRegex(ValueError, "positive"):
            build_records(files, sources, train_per_language=0)


if __name__ == "__main__":
    unittest.main()
