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


def test_cost_details_include_exact_mode_and_pinned_price_metadata() -> None:
    result = (
        pl.DataFrame({"text": ["hello world", None, ""]})
        .lazy()
        .select(
            pl.col("text")
            .tokens.estimate_cost_details(  # ty: ignore[unresolved-attribute]
                model="gpt-5", category="cached_input"
            )
            .alias("estimate")
        )
        .collect(engine="streaming")
    )
    values = result["estimate"].to_list()

    assert values[0]["cost_usd"] == pytest.approx(2 * 0.125 / 1_000_000)
    assert values[1]["cost_usd"] is None
    assert values[2]["cost_usd"] == 0.0
    assert {value["token_count_mode"] for value in values} == {"exact"}
    assert values[0]["model"] == "gpt-5"
    assert values[0]["model_registry_version"] == tokens.MODEL_REGISTRY_VERSION
    assert values[0]["serving_provider"] == "openai"
    assert values[0]["category"] == "cached_input"
    assert values[0]["currency"] == "USD"
    assert values[0]["price_per_unit"] == "0.125"
    assert values[0]["unit_tokens"] == 1_000_000
    assert values[0]["price_source"] == "registry"
    assert values[0]["price_snapshot_date"] == "2026-09-24"
    assert values[0]["price_registry_version"] == tokens.PRICE_REGISTRY_VERSION
    assert values[0]["price_source_url"].startswith("https://developers.openai.com/")


def test_cost_details_label_caller_override_without_inventing_price_snapshot() -> None:
    result = pl.DataFrame({"text": ["hello world"]}).select(
        tokens.estimate_cost_details(
            "text",
            model="gpt-4",
            usd_per_million_tokens=Decimal("3.50"),
        ).alias("estimate")
    )
    value = result["estimate"].item()

    assert value["cost_usd"] == pytest.approx(2 * 3.5 / 1_000_000)
    assert value["token_count_mode"] == "exact"
    assert value["model"] == "gpt-4"
    assert value["price_per_unit"] == "3.50"
    assert value["price_source"] == "caller_override"
    assert value["serving_provider"] is None
    assert value["price_snapshot_date"] is None
    assert value["price_registry_version"] is None
    assert value["price_source_url"] is None


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


def test_explicit_snapshot_date_selects_the_pinned_price() -> None:
    price = tokens.price_info("gpt-5", snapshot_date=tokens.PRICE_SNAPSHOT_DATE)
    result = pl.DataFrame({"text": ["hello world"]}).select(
        tokens.estimate_cost_details(
            "text", model="gpt-5", snapshot_date=tokens.PRICE_SNAPSHOT_DATE
        ).alias("estimate")
    )

    assert price.snapshot_date == tokens.PRICE_SNAPSHOT_DATE
    assert price.registry_version == tokens.PRICE_REGISTRY_VERSION
    assert result["estimate"].item()["price_snapshot_date"] == tokens.PRICE_SNAPSHOT_DATE


@pytest.mark.parametrize("snapshot_date", ["2026-09-23", "2026-09-25"])
def test_price_snapshot_date_boundaries_fail_closed(snapshot_date: str) -> None:
    with pytest.raises(ValueError, match="no price snapshot for date"):
        tokens.price_info("gpt-5", snapshot_date=snapshot_date)
    with pytest.raises(ValueError, match="no price snapshot for date"):
        tokens.estimate_cost("text", model="gpt-5", snapshot_date=snapshot_date)


@pytest.mark.parametrize("snapshot_date", ["2026-9-24", "yesterday", "2026-02-30"])
def test_invalid_snapshot_date_fails_early(snapshot_date: str) -> None:
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        tokens.price_info("gpt-5", snapshot_date=snapshot_date)


def test_snapshot_date_cannot_be_combined_with_caller_rate() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        tokens.estimate_cost(
            "text",
            model="gpt-4",
            usd_per_million_tokens=3.5,
            snapshot_date=tokens.PRICE_SNAPSHOT_DATE,
        )


def test_retired_identifier_outside_pinned_model_registry_fails_early() -> None:
    # OpenAI shut down this snapshot on 2026-03-26. It is not a pinned alias.
    with pytest.raises(ValueError, match="unknown model"):
        tokens.estimate_cost(
            "text",
            model="gpt-4-0314",  # ty: ignore[invalid-argument-type]
            usd_per_million_tokens=30.0,
        )


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
