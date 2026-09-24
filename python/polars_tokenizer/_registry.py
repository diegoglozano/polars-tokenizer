"""Versioned model-to-tokenizer metadata.

This module is intentionally independent from the Rust tokenizer engine. Model
aliases are a provider-facing convenience; the native kernel only receives a
tokenizer identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Tokenizer: TypeAlias = Literal["cl100k_base", "o200k_base"]
Model: TypeAlias = Literal[
    "babbage-002",
    "davinci-002",
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4.1",
    "gpt-4o",
    "gpt-5",
    "o1",
    "o3",
    "o4-mini",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
]

# Changing an alias requires a new registry version. Pricing will have its own
# effective-dated snapshots because tokenizer mappings and prices change on
# independent schedules.
MODEL_REGISTRY_VERSION = "2026-09-24"


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Pinned metadata needed to resolve a model to a tokenizer."""

    owner: str
    tokenizer: Tokenizer


_OPENAI_O200K_MODELS = (
    "gpt-4.1",
    "gpt-4o",
    "gpt-5",
    "o1",
    "o3",
    "o4-mini",
)
_OPENAI_CL100K_MODELS = (
    "babbage-002",
    "davinci-002",
    "gpt-3.5-turbo",
    "gpt-4",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
)

MODEL_REGISTRY: dict[str, ModelInfo] = {
    **{model: ModelInfo(owner="openai", tokenizer="o200k_base") for model in _OPENAI_O200K_MODELS},
    **{
        model: ModelInfo(owner="openai", tokenizer="cl100k_base") for model in _OPENAI_CL100K_MODELS
    },
}


def resolve_model(model: str) -> ModelInfo:
    """Resolve an exact, pinned model alias or raise a useful error."""
    try:
        return MODEL_REGISTRY[model]
    except KeyError:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        msg = (
            f"unknown model {model!r} in model registry {MODEL_REGISTRY_VERSION}; "
            f"supported models: {supported}"
        )
        raise ValueError(msg) from None
