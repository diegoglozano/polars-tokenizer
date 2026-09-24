"""Public Polars expression API."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import polars as pl
from polars.plugins import register_plugin_function

from polars_tokenizer._pricing import (
    BillingCategory,
    UsdPerMillionOverride,
    resolve_price_per_token,
)
from polars_tokenizer._registry import Model, Tokenizer, resolve_model

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
) -> pl.Expr:
    """Estimate raw-text cost in USD using exact counts and pinned pricing.

    This excludes chat wrappers, tools, images, audio, and every other piece of
    request-level accounting. Null input produces null output. A caller may
    replace the registry rate with an explicit USD-per-million-token value.
    """
    price_per_token = resolve_price_per_token(model, category, usd_per_million_tokens)
    return count(expr, model=model).cast(pl.Float64) * float(price_per_token)


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
    ) -> pl.Expr:
        """Estimate raw-text cost in USD from exact local token counts."""
        return estimate_cost(
            self._expr,
            model=model,
            category=category,
            usd_per_million_tokens=usd_per_million_tokens,
        )
