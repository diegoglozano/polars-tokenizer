"""Pinned model pricing metadata used by cost expressions."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, TypeAlias

from polars_tokenizer._registry import Model, resolve_model

BillingCategory: TypeAlias = Literal["cached_input", "input", "output"]
Currency: TypeAlias = Literal["USD"]
PricedModel: TypeAlias = Literal["gpt-5"]

PRICE_REGISTRY_VERSION = "openai-2026-09-24"
_UNIT_TOKENS = 1_000_000


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


_GPT5_PRICES: dict[BillingCategory, Decimal] = {
    "input": Decimal("1.25"),
    "cached_input": Decimal("0.125"),
    "output": Decimal("10.00"),
}


def price_info(model: Model, category: BillingCategory = "input") -> PriceInfo:
    """Return pinned pricing metadata for a supported model and category."""
    # Resolve aliases independently so an unknown model has the model-registry
    # error, while a known but currently unpriced model has a pricing error.
    resolve_model(model)

    if model != "gpt-5":
        msg = (
            f"no price snapshot for model {model!r} in price registry "
            f"{PRICE_REGISTRY_VERSION}; priced models: gpt-5"
        )
        raise ValueError(msg)
    try:
        price = _GPT5_PRICES[category]
    except KeyError:
        supported = ", ".join(_GPT5_PRICES)
        msg = f"unsupported billing category {category!r}; supported categories: {supported}"
        raise ValueError(msg) from None

    return PriceInfo(
        model=model,
        serving_provider="openai",
        category=category,
        currency="USD",
        price_per_unit=price,
        unit_tokens=_UNIT_TOKENS,
        snapshot_date="2026-09-24",
        registry_version=PRICE_REGISTRY_VERSION,
        source_url="https://developers.openai.com/api/docs/models/gpt-5",
    )
