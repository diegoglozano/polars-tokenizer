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
_SUPPORTED = frozenset({"cl100k_base", "o200k_base"})


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
) -> pl.Expr:
    """Return the exact raw-text token count as a UInt32 expression.

    Null input produces null output. Special-token-looking substrings are
    treated as ordinary text; no Unicode normalization is applied.
    """
    resolved = _resolve_tokenizer(tokenizer, model)
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        args=[expr],
        function_name="token_count",
        kwargs={"tokenizer": resolved},
        is_elementwise=True,
    )


def estimate_cost(
    expr: IntoExpr,
    *,
    model: Model,
    category: BillingCategory = "input",
    usd_per_million_tokens: UsdPerMillionOverride | None = None,
    snapshot_date: str | None = None,
) -> pl.Expr:
    """Estimate raw-text cost in USD using exact counts and pinned pricing.

    This excludes chat wrappers, tools, images, audio, and every other piece of
    request-level accounting. Null input produces null output. A caller may
    replace the registry rate with an explicit USD-per-million-token value or
    select an exact bundled snapshot date.
    """
    price_per_token = resolve_price_per_token(
        model, category, usd_per_million_tokens, snapshot_date=snapshot_date
    )
    return count(expr, model=model).cast(pl.Float64) * float(price_per_token)


def estimate_cost_details(
    expr: IntoExpr,
    *,
    model: Model,
    category: BillingCategory = "input",
    usd_per_million_tokens: UsdPerMillionOverride | None = None,
    snapshot_date: str | None = None,
) -> pl.Expr:
    """Return a struct with raw-text cost, count mode, and price provenance.

    The `cost_usd` field is null for null input; metadata fields remain present.
    Caller-supplied rates have no pinned snapshot or known serving provider.
    `snapshot_date` selects one exact bundled price snapshot.
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

    return pl.struct(
        cost_usd=count(expr, model=model).cast(pl.Float64) * float(price_per_token),
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
    ) -> pl.Expr:
        """Return exact token counts for this string expression."""
        return count(self._expr, tokenizer=tokenizer, model=model)

    def estimate_cost(
        self,
        *,
        model: Model,
        category: BillingCategory = "input",
        usd_per_million_tokens: UsdPerMillionOverride | None = None,
        snapshot_date: str | None = None,
    ) -> pl.Expr:
        """Estimate raw-text cost in USD from exact local token counts."""
        return estimate_cost(
            self._expr,
            model=model,
            category=category,
            usd_per_million_tokens=usd_per_million_tokens,
            snapshot_date=snapshot_date,
        )

    def estimate_cost_details(
        self,
        *,
        model: Model,
        category: BillingCategory = "input",
        usd_per_million_tokens: UsdPerMillionOverride | None = None,
        snapshot_date: str | None = None,
    ) -> pl.Expr:
        """Return a struct with cost, token-count mode, and price provenance."""
        return estimate_cost_details(
            self._expr,
            model=model,
            category=category,
            usd_per_million_tokens=usd_per_million_tokens,
            snapshot_date=snapshot_date,
        )
