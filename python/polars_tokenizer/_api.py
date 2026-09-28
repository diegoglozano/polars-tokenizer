"""Public Polars expression API."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TypeAlias

import polars as pl
from polars.plugins import register_plugin_function

from polars_tokenizer._pricing import (
    BillingCategory,
    UsdPerMillionOverride,
    price_info,
    resolve_price_per_token,
)
from polars_tokenizer._registry import MODEL_REGISTRY_VERSION, Model, Tokenizer, resolve_model

IntoExpr: TypeAlias = str | pl.Expr | pl.Series

_PLUGIN_PATH = Path(__file__).parent
_SUPPORTED = frozenset({"cl100k_base", "o200k_base", "p50k_base", "r50k_base"})
_MAX_CACHE_CAPACITY = 65_536


def _validate_tokenizer(tokenizer: str) -> None:
    if tokenizer not in _SUPPORTED:
        supported = ", ".join(sorted(_SUPPORTED))
        msg = f"unsupported tokenizer {tokenizer!r}; supported tokenizers: {supported}"
        raise ValueError(msg)


def _resolve_tokenizer(tokenizer: Tokenizer | None, model: Model | None) -> Tokenizer:
    if tokenizer is not None and model is not None:
        msg = "pass either tokenizer or model, not both"
        raise ValueError(msg)
    if model is not None:
        return resolve_model(model).tokenizer
    resolved = "o200k_base" if tokenizer is None else tokenizer
    _validate_tokenizer(resolved)
    return resolved


def count(
    expr: IntoExpr,
    tokenizer: Tokenizer | None = None,
    *,
    model: Model | None = None,
    cache_capacity: int | None = None,
) -> pl.Expr:
    """Return the exact raw-text token count as a UInt32 expression.

    String, UTF-8 Binary, Categorical, and Enum inputs are supported. Invalid
    non-null Binary UTF-8 raises an error; null input produces null output.
    Special-token-looking substrings are ordinary text; no normalization applies.

    For repeated String or Binary values, ``cache_capacity`` enables a bounded
    FIFO cache of exact counts. It is opt-in because hashing mostly unique
    values can be slower. Categorical and enum columns already count each
    category once, so this setting has no effect on them.
    """
    resolved = _resolve_tokenizer(tokenizer, model)
    if cache_capacity is not None and (
        isinstance(cache_capacity, bool)
        or not isinstance(cache_capacity, int)
        or not 1 <= cache_capacity <= _MAX_CACHE_CAPACITY
    ):
        msg = f"cache_capacity must be an integer between 1 and {_MAX_CACHE_CAPACITY}"
        raise ValueError(msg)
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        args=[expr],
        function_name="token_count",
        kwargs={"tokenizer": resolved, "cache_capacity": cache_capacity},
        is_elementwise=True,
    )


def estimate_cost(
    expr: IntoExpr,
    *,
    model: Model,
    category: BillingCategory = "input",
    usd_per_million_tokens: UsdPerMillionOverride | None = None,
    snapshot_date: str | None = None,
    cache_capacity: int | None = None,
) -> pl.Expr:
    """Estimate raw-text cost in USD using exact counts and pinned pricing.

    This excludes chat wrappers, tools, images, audio, and every other piece of
    request-level accounting. Null input produces null output. A caller may
    replace the registry rate with an explicit USD-per-million-token value or
    select an exact bundled snapshot date.

    ``cache_capacity`` applies the same opt-in text cache as :func:`count`.
    """
    price_per_token = resolve_price_per_token(
        model, category, usd_per_million_tokens, snapshot_date=snapshot_date
    )
    return count(expr, model=model, cache_capacity=cache_capacity).cast(pl.Float64) * float(
        price_per_token
    )


def estimate_cost_details(
    expr: IntoExpr,
    *,
    model: Model,
    category: BillingCategory = "input",
    usd_per_million_tokens: UsdPerMillionOverride | None = None,
    snapshot_date: str | None = None,
    cache_capacity: int | None = None,
) -> pl.Expr:
    """Return a struct with raw-text token count, cost, and price provenance.

    The `cost_usd` field is null for null input; metadata fields remain present.
    Caller-supplied rates have no pinned snapshot or known serving provider.
    `snapshot_date` selects one exact bundled price snapshot.

    ``cache_capacity`` applies the same opt-in text cache as :func:`count`.
    """
    price_per_token = resolve_price_per_token(
        model, category, usd_per_million_tokens, snapshot_date=snapshot_date
    )
    if usd_per_million_tokens is None:
        price = price_info(model, category, snapshot_date=snapshot_date)
        price_per_unit = price.price_per_unit
        serving_provider = price.serving_provider
        selected_snapshot_date = price.snapshot_date
        registry_version = price.registry_version
        source_url = price.source_url
        price_source = "registry"
    else:
        price_per_unit = Decimal(str(usd_per_million_tokens))
        serving_provider = None
        selected_snapshot_date = None
        registry_version = None
        source_url = None
        price_source = "caller_override"

    token_counts = count(expr, model=model, cache_capacity=cache_capacity)
    return pl.struct(
        token_count=token_counts,
        cost_usd=token_counts.cast(pl.Float64) * float(price_per_token),
        token_count_mode=pl.lit("exact"),
        model=pl.lit(model),
        model_registry_version=pl.lit(MODEL_REGISTRY_VERSION),
        serving_provider=pl.lit(serving_provider, dtype=pl.String),
        category=pl.lit(category),
        currency=pl.lit("USD"),
        price_per_unit=pl.lit(str(price_per_unit)),
        unit_tokens=pl.lit(1_000_000),
        price_source=pl.lit(price_source),
        price_snapshot_date=pl.lit(selected_snapshot_date, dtype=pl.String),
        price_registry_version=pl.lit(registry_version, dtype=pl.String),
        price_source_url=pl.lit(source_url, dtype=pl.String),
    )


@pl.api.register_expr_namespace("tokens")
class TokenExprNameSpace:
    """Token analytics methods for :class:`polars.Expr`."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def count(
        self,
        tokenizer: Tokenizer | None = None,
        *,
        model: Model | None = None,
        cache_capacity: int | None = None,
    ) -> pl.Expr:
        """Return exact token counts for this text expression."""
        return count(
            self._expr,
            tokenizer=tokenizer,
            model=model,
            cache_capacity=cache_capacity,
        )

    def estimate_cost(
        self,
        *,
        model: Model,
        category: BillingCategory = "input",
        usd_per_million_tokens: UsdPerMillionOverride | None = None,
        snapshot_date: str | None = None,
        cache_capacity: int | None = None,
    ) -> pl.Expr:
        """Estimate raw-text cost in USD from exact local token counts."""
        return estimate_cost(
            self._expr,
            model=model,
            category=category,
            usd_per_million_tokens=usd_per_million_tokens,
            snapshot_date=snapshot_date,
            cache_capacity=cache_capacity,
        )

    def estimate_cost_details(
        self,
        *,
        model: Model,
        category: BillingCategory = "input",
        usd_per_million_tokens: UsdPerMillionOverride | None = None,
        snapshot_date: str | None = None,
        cache_capacity: int | None = None,
    ) -> pl.Expr:
        """Return a struct with token count, cost, and price provenance."""
        return estimate_cost_details(
            self._expr,
            model=model,
            category=category,
            usd_per_million_tokens=usd_per_million_tokens,
            snapshot_date=snapshot_date,
            cache_capacity=cache_capacity,
        )
