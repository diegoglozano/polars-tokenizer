from __future__ import annotations

import polars as pl
import polars_tokenizer as tokens
import pytest
import tiktoken
from hypothesis import given, settings
from hypothesis import strategies as st
from polars_tokenizer import Tokenizer

TOKENIZERS: tuple[Tokenizer, ...] = (
    "cl100k_base",
    "o200k_base",
    "p50k_base",
    "r50k_base",
)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
@pytest.mark.parametrize("cache_capacity", [None, 2])
def test_binary_utf8_matches_string_counts(
    tokenizer: Tokenizer, cache_capacity: int | None
) -> None:
    texts = ["", "hello world", "你好世界", "👋🏽🌍", "<|endoftext|>", None]
    binary = pl.Series(
        "text", [text.encode() if text is not None else None for text in texts], dtype=pl.Binary
    )
    strings = pl.Series("text", texts, dtype=pl.String)
    actual = pl.DataFrame(binary).select(
        tokens.count("text", tokenizer, cache_capacity=cache_capacity)
    )
    expected = pl.DataFrame(strings).select(tokens.count("text", tokenizer))
    assert actual.schema == expected.schema == {"text": pl.UInt32}
    assert actual.to_series().to_list() == expected.to_series().to_list()


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_binary_nulls_ignore_invalid_retained_bytes(tokenizer: Tokenizer) -> None:
    original = pl.Series("text", [b"\xff", b"hello", b"\xfe", b""], dtype=pl.Binary)
    values = original.set(pl.Series([True, False, True, False]), None)
    assert values.to_list() == [None, b"hello", None, b""]
    actual = (
        pl.DataFrame(values)
        .lazy()
        .select(tokens.count("text", tokenizer, cache_capacity=2))
        .collect(engine="streaming")
        .to_series()
    )
    assert actual.to_list() == [None, 1, None, 0]


@pytest.mark.parametrize("cache_capacity", [None, 2])
def test_all_null_binary_with_invalid_retained_bytes(cache_capacity: int | None) -> None:
    original = pl.Series("text", [b"\xff", b"\xfe"] * 32, dtype=pl.Binary)
    values = original.set(pl.Series([True] * len(original)), None)
    actual = pl.DataFrame(values).select(tokens.count("text", cache_capacity=cache_capacity))
    assert actual.to_series().to_list() == [None] * len(values)


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
@pytest.mark.parametrize("cache_capacity", [None, 2])
def test_zero_byte_binary_columns(tokenizer: Tokenizer, cache_capacity: int | None) -> None:
    for values, expected in (
        ([b"", b""], [0, 0]),
        ([None, None], [None, None]),
        ([b"", None, b""], [0, None, 0]),
        ([], []),
    ):
        first = pl.Series("text", values[:1], dtype=pl.Binary)
        second = pl.Series("text", values[1:], dtype=pl.Binary)
        frame = pl.DataFrame(first.append(second))
        actual = frame.select(tokens.count("text", tokenizer, cache_capacity=cache_capacity))
        assert actual.to_series().to_list() == expected
        assert actual.schema == {"text": pl.UInt32}


@pytest.mark.parametrize("cache_capacity", [None, 2])
def test_invalid_non_null_binary_errors(cache_capacity: int | None) -> None:
    frame = pl.DataFrame({"text": pl.Series([b"hello", b"\xff"], dtype=pl.Binary)})
    with pytest.raises(pl.exceptions.ComputeError, match="Binary input contains invalid UTF-8"):
        frame.select(tokens.count("text", cache_capacity=cache_capacity))


def test_parallel_multichunk_binary_slice_matches_reference() -> None:
    texts = [
        None if index % 17 == 0 else f"row {index}: hello 世界 👋🏽 " * 7 for index in range(10_000)
    ]
    first = pl.Series(
        "text",
        [text.encode() if text is not None else None for text in texts[:5_000]],
        dtype=pl.Binary,
    )
    second = pl.Series(
        "text",
        [text.encode() if text is not None else None for text in texts[5_000:]],
        dtype=pl.Binary,
    )
    values = first.append(second).slice(9, 9_980)
    assert values.n_chunks() > 1
    assert sum(len(value) for value in values if value is not None) > 512 * 1024
    expected = (
        pl.DataFrame({"text": values.cast(pl.String)})
        .select(tokens.count("text"))
        .to_series()
        .to_list()
    )
    frame = pl.DataFrame(values)
    for cache_capacity in (None, 32):
        actual = frame.select(tokens.count("text", cache_capacity=cache_capacity)).to_series()
        assert actual.to_list() == expected


def test_parallel_binary_invalid_utf8_errors() -> None:
    values = [b"hello world " * 16 for _ in range(8_000)]
    values[7_777] = b"\xff"
    frame = pl.DataFrame({"text": pl.Series(values, dtype=pl.Binary)})
    with pytest.raises(pl.exceptions.ComputeError, match="Binary input contains invalid UTF-8"):
        frame.select(tokens.count("text"))


@pytest.mark.parametrize("cache_capacity", [None, 32])
def test_parallel_binary_ignores_invalid_bytes_under_nulls(cache_capacity: int | None) -> None:
    valid_text = b"hello world " * 16
    original = pl.Series(
        "text",
        [b"\xff" if index % 2 == 0 else valid_text for index in range(8_000)],
        dtype=pl.Binary,
    )
    values = original.set(pl.Series([index % 2 == 0 for index in range(8_000)]), None)
    assert sum(len(value) for value in values if value is not None) > 512 * 1024
    expected_count = len(tiktoken.get_encoding("o200k_base").encode(valid_text.decode()))
    actual = (
        pl.DataFrame(values)
        .select(tokens.count("text", cache_capacity=cache_capacity))
        .to_series()
        .to_list()
    )
    assert actual == [None if index % 2 == 0 else expected_count for index in range(8_000)]


def test_sparse_empty_binary_ignores_large_invalid_null_views() -> None:
    original = pl.Series(
        "text",
        [b"" if index % 1_000 == 0 else b"\xff" * 128 for index in range(16_000)],
        dtype=pl.Binary,
    )
    values = original.set(pl.Series([index % 1_000 != 0 for index in range(16_000)]), None)
    actual = pl.DataFrame(values).select(tokens.count("text")).to_series().to_list()
    assert actual == [0 if index % 1_000 == 0 else None for index in range(16_000)]


def test_binary_cost_details_preserve_count_and_nulls() -> None:
    values = pl.Series("text", [b"hello world", None, "你好".encode()], dtype=pl.Binary)
    details = pl.DataFrame(values).select(tokens.estimate_cost_details("text", model="gpt-5"))
    assert details.to_series().struct.field("token_count").to_list() == [2, None, 1]


@settings(max_examples=100, deadline=None)
@given(st.text(max_size=512))
def test_arbitrary_utf8_binary_matches_reference(text: str) -> None:
    frame = pl.DataFrame({"text": pl.Series([text.encode()], dtype=pl.Binary)})
    actual = frame.select(tokens.count("text")).item()
    expected = len(tiktoken.get_encoding("o200k_base").encode(text, disallowed_special=()))
    assert actual == expected
