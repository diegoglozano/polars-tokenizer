from __future__ import annotations

import polars as pl
import polars_tokenizer as tokens
import pytest
import tiktoken
from hypothesis import given, settings
from hypothesis import strategies as st

ENCODING = tiktoken.get_encoding("o200k_base")


def reference_count(text: str) -> int:
    return len(ENCODING.encode(text, disallowed_special=()))


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello world",
        "hello\0world",
        "line one\r\nline two",
        "e\N{COMBINING ACUTE ACCENT}",
        "\N{ZERO WIDTH SPACE}\N{ZERO WIDTH JOINER}",
        "你好，世界",
        "مرحبا بالعالم",
        "👋🏽🌍🧑‍💻",
        "<|endoftext|>",
        '{"number": 1234567890, "ok": true}',
        'fn main() { println!("hello"); }',
    ],
)
def test_exact_reference_cases(text: str) -> None:
    result = pl.DataFrame({"text": [text]}).select(
        pl.col("text").tokens.count("o200k_base")  # ty: ignore[unresolved-attribute]
    )
    assert result.item() == reference_count(text)
    assert result.schema["text"] == pl.UInt32


def test_null_and_empty() -> None:
    result = pl.DataFrame({"text": ["hello world", None, ""]}).select(
        tokens.count("text").alias("count")
    )
    assert result["count"].to_list() == [2, None, 0]


def test_chunked_input() -> None:
    first = pl.Series("text", ["alpha", None])
    second = pl.Series("text", ["βήτα", "omega"])
    frame = pl.DataFrame(first.append(second))
    expected = [reference_count(value) if value is not None else None for value in frame["text"]]
    assert frame.select(tokens.count("text")).to_series().to_list() == expected


def test_lazy_streaming() -> None:
    frame = pl.DataFrame({"text": ["one", "two words", None, ""]})
    result = frame.lazy().select(tokens.count("text")).collect(engine="streaming")
    assert result.to_series().to_list() == [1, 2, None, 0]


def test_expression_composes() -> None:
    frame = pl.DataFrame({"title": ["hello"], "body": ["world"]})
    result = frame.select(
        (
            pl.col("title").tokens.count()  # ty: ignore[unresolved-attribute]
            + pl.col("body").tokens.count()  # ty: ignore[unresolved-attribute]
        ).alias("combined")
    )
    assert result.item() == 2


def test_large_parallel_batch_matches_reference_and_is_deterministic() -> None:
    values = [
        None if index % 17 == 0 else f"row {index}: hello 世界 👋🏽 " * (index % 7 + 1)
        for index in range(8_000)
    ]
    frame = pl.DataFrame({"text": values})
    expected = [reference_count(value) if value is not None else None for value in values]

    for _ in range(3):
        actual = frame.select(tokens.count("text")).to_series().to_list()
        assert actual == expected


def test_byte_balancing_handles_one_large_row() -> None:
    values = ["small", ("one very long row " * 20_000), None, "tail"]
    frame = pl.DataFrame({"text": values})
    expected = [reference_count(value) if value is not None else None for value in values]
    assert frame.select(tokens.count("text")).to_series().to_list() == expected


def test_group_by_context() -> None:
    frame = pl.DataFrame(
        {"group": ["a", "a", "b", "b"], "text": ["hello", "two words", None, "你好"]}
    )
    result = frame.group_by("group", maintain_order=True).agg(tokens.count("text")).to_dicts()
    assert result == [
        {"group": "a", "text": [1, 2]},
        {"group": "b", "text": [None, reference_count("你好")]},
    ]


def test_categorical_counts_dictionary_values_once_semantically() -> None:
    values = ["hello world", "你好", None, "hello world", "", "你好"]
    frame = pl.DataFrame({"text": values}).with_columns(pl.col("text").cast(pl.Categorical))
    expected = [reference_count(value) if value is not None else None for value in values]

    result = frame.select(tokens.count("text"))
    assert result.to_series().to_list() == expected
    assert result.schema["text"] == pl.UInt32


def test_enum_with_unused_categories() -> None:
    values = ["alpha", None, "beta", "alpha"]
    dtype = pl.Enum(["unused", "alpha", "beta", "also unused"])
    frame = pl.DataFrame({"text": pl.Series(values, dtype=dtype)})
    expected = [reference_count(value) if value is not None else None for value in values]

    assert frame.select(tokens.count("text")).to_series().to_list() == expected


def test_categorical_lazy_streaming() -> None:
    values = ["repeat", "repeat", None, "different"]
    frame = pl.DataFrame({"text": values}).with_columns(pl.col("text").cast(pl.Categorical))
    expected = [reference_count(value) if value is not None else None for value in values]
    result = frame.lazy().select(tokens.count("text")).collect(engine="streaming")
    assert result.to_series().to_list() == expected


def test_wide_categorical_ids_match_reference() -> None:
    values = [f"category-{index}" for index in range(66_000)]
    frame = pl.DataFrame({"text": values}).with_columns(pl.col("text").cast(pl.Categorical))
    expected = [reference_count(value) for value in values]
    assert frame.select(tokens.count("text")).to_series().to_list() == expected


def test_sparse_enum_mapping_uses_only_present_values() -> None:
    categories = [f"category-{index}" for index in range(2_048)]
    values = [categories[3], None, categories[-1], categories[3]]
    frame = pl.DataFrame({"text": pl.Series(values, dtype=pl.Enum(categories))})
    expected = [reference_count(value) if value is not None else None for value in values]
    assert frame.select(tokens.count("text")).to_series().to_list() == expected


def test_unsupported_tokenizer_fails_early() -> None:
    with pytest.raises(ValueError, match="unsupported tokenizer"):
        tokens.count(
            "text",
            tokenizer="cl100k_base",  # ty: ignore[invalid-argument-type]
        )


def test_non_string_column_errors() -> None:
    with pytest.raises(
        pl.exceptions.ComputeError, match="expected `String`, `Categorical`, or `Enum`"
    ):
        pl.DataFrame({"value": [1, 2]}).select(tokens.count("value"))


@settings(max_examples=500, deadline=None)
@given(st.text(max_size=2_048))
def test_arbitrary_unicode_matches_reference(text: str) -> None:
    actual = pl.DataFrame({"text": [text]}).select(tokens.count("text")).item()
    assert actual == reference_count(text)
