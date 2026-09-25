"""Pinned model pricing metadata used by cost expressions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal, TypeAlias

from polars_tokenizer._registry import Model, resolve_model

BillingCategory: TypeAlias = Literal["cached_input", "input", "output"]
Currency: TypeAlias = Literal["USD"]
PricedModel: TypeAlias = Literal["gpt-4.1", "gpt-4o", "gpt-5"]
UsdPerMillionOverride: TypeAlias = Decimal | int | float

PRICE_SNAPSHOT_DATE = "2026-09-25"
PRICE_REGISTRY_VERSION = f"openai-{PRICE_SNAPSHOT_DATE}"
_UNIT_TOKENS = 1_000_000
_BILLING_CATEGORIES: tuple[BillingCategory, ...] = ("input", "cached_input", "output")


@dataclass(frozen=True, slots=True)
class PriceInfo:
    """One immutable price from a dated registry snapshot."""

    model: str
    serving_provider: str
    category: BillingCategory
    currency: Currency
    price_per_unit: Decimal
    unit_tokens: int
    snapshot_date: str
    registry_version: str
    source_url: str

    @property
    def price_per_token(self) -> Decimal:
        """Return the exact decimal price for one token."""
        return self.price_per_unit / self.unit_tokens


_PRICE_SNAPSHOTS: dict[str, dict[str, dict[BillingCategory, Decimal]]] = {
    "2026-09-24": {
        "gpt-5": {
            "input": Decimal("1.25"),
            "cached_input": Decimal("0.125"),
            "output": Decimal("10.00"),
        }
    },
    "2026-09-25": {
        "gpt-4.1": {
            "input": Decimal("2.00"),
            "cached_input": Decimal("0.50"),
            "output": Decimal("8.00"),
        },
        "gpt-4o": {
            "input": Decimal("2.50"),
            "cached_input": Decimal("1.25"),
            "output": Decimal("10.00"),
        },
        "gpt-5": {
            "input": Decimal("1.25"),
            "cached_input": Decimal("0.125"),
            "output": Decimal("10.00"),
        },
    },
}
_PRICE_SOURCE_URLS = {
    "gpt-4.1": "https://developers.openai.com/api/docs/models/gpt-4.1",
    "gpt-4o": "https://developers.openai.com/api/docs/models/gpt-4o",
    "gpt-5": "https://developers.openai.com/api/docs/models/gpt-5",
}


def _resolve_snapshot_date(snapshot_date: str | None) -> str:
    if snapshot_date is None:
        return PRICE_SNAPSHOT_DATE
    try:
        parsed = date.fromisoformat(snapshot_date)
    except (TypeError, ValueError):
        msg = "snapshot_date must be an ISO date in YYYY-MM-DD format"
        raise ValueError(msg) from None
    if parsed.isoformat() != snapshot_date:
        msg = "snapshot_date must be an ISO date in YYYY-MM-DD format"
        raise ValueError(msg)
    if snapshot_date not in _PRICE_SNAPSHOTS:
        available = ", ".join(sorted(_PRICE_SNAPSHOTS))
        msg = f"no price snapshot for date {snapshot_date!r}; available snapshot dates: {available}"
        raise ValueError(msg)
    return snapshot_date


def price_info(
    model: Model,
    category: BillingCategory = "input",
    *,
    snapshot_date: str | None = None,
) -> PriceInfo:
    """Return pricing metadata from an exact pinned snapshot date."""
    # Resolve aliases independently so an unknown model has the model-registry
    # error, while a known but currently unpriced model has a pricing error.
    resolve_model(model)
    selected_date = _resolve_snapshot_date(snapshot_date)

    snapshot = _PRICE_SNAPSHOTS[selected_date]
    if model not in snapshot:
        priced_models = ", ".join(sorted(snapshot))
        msg = (
            f"no price snapshot for model {model!r} in price registry "
            f"openai-{selected_date}; priced models: {priced_models}"
        )
        raise ValueError(msg)
    try:
        price = snapshot[model][category]
    except KeyError:
        supported = ", ".join(_BILLING_CATEGORIES)
        msg = f"unsupported billing category {category!r}; supported categories: {supported}"
        raise ValueError(msg) from None

    return PriceInfo(
        model=model,
        serving_provider="openai",
        category=category,
        currency="USD",
        price_per_unit=price,
        unit_tokens=_UNIT_TOKENS,
        snapshot_date=selected_date,
        registry_version=f"openai-{selected_date}",
        source_url=_PRICE_SOURCE_URLS[model],
    )


def resolve_price_per_token(
    model: Model,
    category: BillingCategory,
    usd_per_million_tokens: UsdPerMillionOverride | None,
    *,
    snapshot_date: str | None = None,
) -> Decimal:
    """Resolve a pinned or caller-supplied rate to USD per token."""
    if usd_per_million_tokens is None:
        return price_info(model, category, snapshot_date=snapshot_date).price_per_token

    resolve_model(model)
    if snapshot_date is not None:
        msg = "snapshot_date cannot be combined with usd_per_million_tokens"
        raise ValueError(msg)
    if category not in _BILLING_CATEGORIES:
        supported = ", ".join(_BILLING_CATEGORIES)
        msg = f"unsupported billing category {category!r}; supported categories: {supported}"
        raise ValueError(msg)
    if isinstance(usd_per_million_tokens, bool):
        msg = "usd_per_million_tokens must be a finite, non-negative number"
        raise ValueError(msg)
    try:
        override = Decimal(str(usd_per_million_tokens))
    except (InvalidOperation, ValueError):
        msg = "usd_per_million_tokens must be a finite, non-negative number"
        raise ValueError(msg) from None
    if not override.is_finite() or override < 0:
        msg = "usd_per_million_tokens must be a finite, non-negative number"
        raise ValueError(msg)
    return override / _UNIT_TOKENS
