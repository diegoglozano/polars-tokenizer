from __future__ import annotations

from decimal import Decimal

import polars as pl
import polars_tokenizer as tokens
import pytest


@pytest.mark.parametrize(
    ("category", "price_per_million"),
    [
        ("input", 1.25),
        ("cached_input", 0.125),
        ("output", 10.0),
    ],
)
def test_gpt5_cost_categories(category: str, price_per_million: float) -> None:
    frame = pl.DataFrame({"text": ["hello world", None, ""]})
    result = frame.select(
        tokens.estimate_cost(
            "text",
            model="gpt-5",
            category=category,  # ty: ignore[invalid-argument-type]
        ).alias("cost_usd")
    )

    assert result.schema["cost_usd"] == pl.Float64
    assert result["cost_usd"].to_list() == pytest.approx(
        [2 * price_per_million / 1_000_000, None, 0.0]
    )


def test_cost_expr_namespace() -> None:
    result = (
        pl.DataFrame({"text": ["hello world"]})
        .lazy()
        .select(
            pl.col("text")
            .tokens.estimate_cost(  # ty: ignore[unresolved-attribute]
                model="gpt-5", category="output"
            )
            .alias("cost_usd")
        )
        .collect(engine="streaming")
    )
    assert result.item() == pytest.approx(2 * 10.0 / 1_000_000)


def test_price_metadata_is_pinned_and_inspectable() -> None:
    price = tokens.price_info("gpt-5", "cached_input")

    assert price.model == "gpt-5"
    assert price.serving_provider == "openai"
    assert price.category == "cached_input"
    assert price.currency == "USD"
    assert price.price_per_unit == Decimal("0.125")
    assert price.unit_tokens == 1_000_000
    assert price.price_per_token == Decimal("0.000000125")
    assert price.snapshot_date == "2026-09-24"
    assert price.registry_version == tokens.PRICE_REGISTRY_VERSION
    assert price.source_url.startswith("https://developers.openai.com/")


def test_known_model_without_price_snapshot_fails_early() -> None:
    with pytest.raises(ValueError, match="no price snapshot"):
        tokens.estimate_cost(
            "text",
            model="gpt-4",
        )


def test_caller_price_override_supports_known_unpriced_model() -> None:
    result = pl.DataFrame({"text": ["hello world", None]}).select(
        tokens.estimate_cost(
            "text",
            model="gpt-4",
            category="input",
            usd_per_million_tokens=Decimal("3.50"),
        ).alias("cost_usd")
    )
    assert result["cost_usd"].to_list() == pytest.approx([2 * 3.5 / 1_000_000, None])


@pytest.mark.parametrize(
    "rate",
    [-1, float("nan"), float("inf"), Decimal("NaN"), True],
)
def test_invalid_caller_price_override_fails_early(rate: object) -> None:
    with pytest.raises(ValueError, match="finite, non-negative"):
        tokens.estimate_cost(
            "text",
            model="gpt-5",
            usd_per_million_tokens=rate,  # ty: ignore[invalid-argument-type]
        )


def test_unknown_model_with_price_override_still_fails() -> None:
    with pytest.raises(ValueError, match=tokens.MODEL_REGISTRY_VERSION):
        tokens.estimate_cost(
            "text",
            model="future-model",  # ty: ignore[invalid-argument-type]
            usd_per_million_tokens=1.0,
        )


def test_unknown_model_uses_model_registry_error() -> None:
    with pytest.raises(ValueError, match=tokens.MODEL_REGISTRY_VERSION):
        tokens.estimate_cost(
            "text",
            model="future-model",  # ty: ignore[invalid-argument-type]
        )


def test_unknown_billing_category_fails_early() -> None:
    with pytest.raises(ValueError, match="unsupported billing category"):
        tokens.estimate_cost(
            "text",
            model="gpt-5",
            category="training",  # ty: ignore[invalid-argument-type]
        )
