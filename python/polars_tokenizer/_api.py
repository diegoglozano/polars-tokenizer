"""Public Polars expression API."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

import polars as pl
from polars.plugins import register_plugin_function

Tokenizer: TypeAlias = Literal["o200k_base"]
IntoExpr: TypeAlias = str | pl.Expr | pl.Series

_PLUGIN_PATH = Path(__file__).parent
_SUPPORTED = frozenset({"o200k_base"})


def _validate_tokenizer(tokenizer: str) -> None:
    if tokenizer not in _SUPPORTED:
        supported = ", ".join(sorted(_SUPPORTED))
        msg = f"unsupported tokenizer {tokenizer!r}; supported tokenizers: {supported}"
        raise ValueError(msg)


def count(expr: IntoExpr, tokenizer: Tokenizer = "o200k_base") -> pl.Expr:
    """Return the exact raw-text token count as a UInt32 expression.

    Null input produces null output. Special-token-looking substrings are
    treated as ordinary text; no Unicode normalization is applied.
    """
    _validate_tokenizer(tokenizer)
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        args=[expr],
        function_name="token_count",
        kwargs={"tokenizer": tokenizer},
        is_elementwise=True,
    )


@pl.api.register_expr_namespace("tokens")
class TokenExprNameSpace:
    """Token analytics methods for :class:`polars.Expr`."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def count(self, tokenizer: Tokenizer = "o200k_base") -> pl.Expr:
        """Return exact token counts for this string expression."""
        return count(self._expr, tokenizer=tokenizer)
