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
ENCODINGS = {name: tiktoken.get_encoding(name) for name in TOKENIZERS}


def reference_count(text: str, tokenizer: Tokenizer = "o200k_base") -> int:
    return len(ENCODINGS[tokenizer].encode(text, disallowed_special=()))


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
@pytest.mark.parametrize("tokenizer", ["cl100k_base", "p50k_base", "r50k_base"])
def test_other_encodings_exact_reference_cases(text: str, tokenizer: Tokenizer) -> None:
    result = pl.DataFrame({"text": [text]}).select(
        pl.col("text").tokens.count(tokenizer)  # ty: ignore[unresolved-attribute]
    )
    assert result.item() == reference_count(text, tokenizer)
    assert result.schema["text"] == pl.UInt32


def test_legacy_code_indentation_uses_distinct_vocabularies() -> None:
    text = "def f():\n" + " " * 24 + "return 1"
    frame = pl.DataFrame({"text": [text]})
    p50k = frame.select(tokens.count("text", "p50k_base")).item()
    r50k = frame.select(tokens.count("text", "r50k_base")).item()
    assert p50k == reference_count(text, "p50k_base") == 7
    assert r50k == reference_count(text, "r50k_base") == 29


def test_null_and_empty() -> None:
    result = pl.DataFrame({"text": ["hello world", None, ""]}).select(
        tokens.count("text").alias("count")
    )
    assert result["count"].to_list() == [2, None, 0]


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
@pytest.mark.parametrize("cache_capacity", [None, 2])
def test_zero_byte_string_columns(tokenizer: Tokenizer, cache_capacity: int | None) -> None:
    for values, expected in (
        (["", "", ""], [0, 0, 0]),
        ([None, None, None], [None, None, None]),
        (["", None, ""], [0, None, 0]),
        ([], []),
    ):
        first = pl.Series("text", values[:1], dtype=pl.String)
        second = pl.Series("text", values[1:], dtype=pl.String)
        frame = pl.DataFrame(first.append(second))
        actual = frame.select(
            tokens.count("text", tokenizer, cache_capacity=cache_capacity)
        ).to_series()
        assert actual.to_list() == expected
        assert actual.dtype == pl.UInt32


@pytest.mark.parametrize("value", ["", None])
def test_zero_byte_string_columns_lazy_streaming(value: str | None) -> None:
    frame = pl.DataFrame({"text": pl.Series([value] * 64, dtype=pl.String)})
    result = frame.lazy().select(tokens.count("text")).collect(engine="streaming")
    assert result.to_series().to_list() == ([0] if value == "" else [None]) * 64


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_non_null_multichunk_slice(tokenizer: Tokenizer) -> None:
    first = pl.Series("text", ["", "hello world", "你好，世界"])
    second = pl.Series("text", ["👋🏽", "line one\nline two", "<|endoftext|>"])
    values = first.append(second).slice(1, 4)
    assert values.null_count() == 0
    expected = [reference_count(value, tokenizer) for value in values]
    actual = pl.DataFrame(values).select(tokens.count("text", tokenizer=tokenizer)).to_series()
    assert actual.to_list() == expected


def test_chunked_input() -> None:
    first = pl.Series("text", ["alpha", None])
    second = pl.Series("text", ["βήτα", "omega"])
    frame = pl.DataFrame(first.append(second))
    expected = [reference_count(value) if value is not None else None for value in frame["text"]]
    assert frame.select(tokens.count("text")).to_series().to_list() == expected


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
@pytest.mark.parametrize("capacity", [1, 2, 16])
def test_bounded_cache_matches_reference_with_nulls_and_evictions(
    tokenizer: Tokenizer, capacity: int
) -> None:
    values = ["hello world", "你好，世界", None, "hello world", "", "👋🏽🌍"] * 8
    first = pl.Series("text", values[:17])
    second = pl.Series("text", values[17:])
    frame = pl.DataFrame(first.append(second))
    expected = [
        reference_count(value, tokenizer) if value is not None else None for value in values
    ]

    result = frame.select(tokens.count("text", tokenizer, cache_capacity=capacity))
    assert result.to_series().to_list() == expected


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_bounded_cache_parallel_matches_uncached(tokenizer: Tokenizer) -> None:
    unique = [f"value {index:04}: " + "hello world 👋🏽 " * 8 for index in range(64)]
    values = [None if index % 101 == 0 else unique[index % len(unique)] for index in range(8_000)]
    frame = pl.DataFrame({"text": values})
    assert frame["text"].str.len_bytes().sum() > 512 * 1024

    uncached = frame.select(tokens.count("text", tokenizer)).to_series().to_list()
    cached = frame.select(tokens.count("text", tokenizer, cache_capacity=256)).to_series()
    assert cached.to_list() == uncached


@pytest.mark.parametrize("capacity", [0, -1, 65_537, True, 2.5, "256"])
def test_invalid_cache_capacity_fails_early(capacity: object) -> None:
    with pytest.raises(ValueError, match="cache_capacity must be an integer"):
        tokens.count("text", cache_capacity=capacity)  # ty: ignore[invalid-argument-type]


def test_bounded_cache_has_no_effect_on_categorical_or_enum() -> None:
    values = ["hello world", None, "你好，世界", "hello world"]
    for dtype in (pl.Categorical, pl.Enum(["unused", "hello world", "你好，世界"])):
        frame = pl.DataFrame({"text": pl.Series(values, dtype=dtype)})
        actual = frame.select(tokens.count("text", cache_capacity=1)).to_series().to_list()
        expected = [reference_count(value) if value is not None else None for value in values]
        assert actual == expected


def test_bounded_cache_expr_namespace_streaming() -> None:
    values = ["hello world", "hello world", None, "你好，世界"]
    frame = pl.DataFrame({"text": values})
    actual = (
        frame.lazy()
        .select(pl.col("text").tokens.count(cache_capacity=2))  # ty: ignore[unresolved-attribute]
        .collect(engine="streaming")
        .to_series()
        .to_list()
    )
    expected = [reference_count(value) if value is not None else None for value in values]
    assert actual == expected


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


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_non_null_categorical_multichunk_slice(tokenizer: Tokenizer) -> None:
    with pl.StringCache():
        first = pl.Series("text", ["hello world", "你好", "hello world"]).cast(pl.Categorical)
        second = pl.Series("text", ["", "👋🏽", "你好"]).cast(pl.Categorical)
        values = first.append(second).slice(1, 4)
    assert values.n_chunks() > 1
    assert values.null_count() == 0
    expected = [reference_count(value, tokenizer) for value in values]
    actual = pl.DataFrame(values).select(tokens.count("text", tokenizer=tokenizer)).to_series()
    assert actual.to_list() == expected


def test_enum_with_unused_categories() -> None:
    values = ["alpha", None, "beta", "alpha"]
    dtype = pl.Enum(["unused", "alpha", "beta", "also unused"])
    frame = pl.DataFrame({"text": pl.Series(values, dtype=dtype)})
    expected = [reference_count(value) if value is not None else None for value in values]

    assert frame.select(tokens.count("text")).to_series().to_list() == expected


def test_non_null_enum_with_unused_categories() -> None:
    values = ["beta", "alpha", "beta", "alpha"]
    dtype = pl.Enum(["unused", "alpha", "beta", "also unused"])
    frame = pl.DataFrame({"text": pl.Series(values, dtype=dtype)})
    expected = [reference_count(value) for value in values]
    assert frame.select(tokens.count("text")).to_series().to_list() == expected


def test_categorical_lazy_streaming() -> None:
    values = ["repeat", "repeat", None, "different"]
    frame = pl.DataFrame({"text": values}).with_columns(pl.col("text").cast(pl.Categorical))
    expected = [reference_count(value) if value is not None else None for value in values]
    result = frame.lazy().select(tokens.count("text")).collect(engine="streaming")
    assert result.to_series().to_list() == expected


def test_cl100k_categorical_lazy_streaming() -> None:
    values = ["repeat", "repeat", None, "different", "你好，世界"]
    frame = pl.DataFrame({"text": values}).with_columns(pl.col("text").cast(pl.Categorical))
    expected = [
        reference_count(value, "cl100k_base") if value is not None else None for value in values
    ]
    result = (
        frame.lazy()
        .select(tokens.count("text", tokenizer="cl100k_base"))
        .collect(engine="streaming")
    )
    assert result.to_series().to_list() == expected


@pytest.mark.parametrize(
    ("model", "tokenizer"),
    [
        ("gpt-5", "o200k_base"),
        ("gpt-4o", "o200k_base"),
        ("gpt-4", "cl100k_base"),
        ("text-embedding-3-small", "cl100k_base"),
    ],
)
def test_model_alias_matches_pinned_tokenizer(model: str, tokenizer: str) -> None:
    values = ["hello world", "你好，世界", None, ""]
    frame = pl.DataFrame({"text": values})

    by_model = (
        frame.select(
            tokens.count(
                "text",
                model=model,  # ty: ignore[invalid-argument-type]
            )
        )
        .to_series()
        .to_list()
    )
    by_tokenizer = (
        frame.select(
            tokens.count(
                "text",
                tokenizer=tokenizer,  # ty: ignore[invalid-argument-type]
            )
        )
        .to_series()
        .to_list()
    )
    assert by_model == by_tokenizer


def test_model_alias_expr_namespace() -> None:
    result = pl.DataFrame({"text": ["hello world"]}).select(
        pl.col("text").tokens.count(model="gpt-5")  # ty: ignore[unresolved-attribute]
    )
    assert result.item() == reference_count("hello world")


def test_tokenizer_and_model_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either tokenizer or model"):
        tokens.count("text", tokenizer="o200k_base", model="gpt-5")


def test_unknown_model_fails_early_with_registry_version() -> None:
    with pytest.raises(ValueError, match=tokens.MODEL_REGISTRY_VERSION):
        tokens.count(
            "text",
            model="future-model",  # ty: ignore[invalid-argument-type]
        )


def test_wide_categorical_ids_match_reference() -> None:
    values = [f"category-{index}" for index in range(66_000)]
    frame = pl.DataFrame({"text": values}).with_columns(pl.col("text").cast(pl.Categorical))
    expected = [reference_count(value) for value in values]
    assert frame.select(tokens.count("text")).to_series().to_list() == expected


@pytest.mark.parametrize("tokenizer", TOKENIZERS)
def test_many_short_categorical_values_match_reference(tokenizer: Tokenizer) -> None:
    categories = [f"key-{index:04}" for index in range(5_000)]
    values = [*categories, None, *categories[::17]]
    frame = pl.DataFrame({"text": values}).with_columns(pl.col("text").cast(pl.Categorical))
    expected = [
        reference_count(value, tokenizer) if value is not None else None for value in values
    ]
    assert frame.select(tokens.count("text", tokenizer=tokenizer)).to_series().to_list() == expected


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
            tokenizer="o200k_harmony",  # ty: ignore[invalid-argument-type]
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


@settings(max_examples=250, deadline=None)
@given(st.text(max_size=2_048))
def test_cl100k_arbitrary_unicode_matches_reference(text: str) -> None:
    actual = (
        pl.DataFrame({"text": [text]}).select(tokens.count("text", tokenizer="cl100k_base")).item()
    )
    assert actual == reference_count(text, "cl100k_base")


@pytest.mark.parametrize("tokenizer", ["p50k_base", "r50k_base"])
@settings(max_examples=125, deadline=None)
@given(text=st.text(max_size=2_048))
def test_legacy_arbitrary_unicode_matches_reference(text: str, tokenizer: Tokenizer) -> None:
    actual = pl.DataFrame({"text": [text]}).select(tokens.count("text", tokenizer)).item()
    assert actual == reference_count(text, tokenizer)
