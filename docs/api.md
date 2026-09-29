# API reference

The public functions live in `polars_tokenizer`. Importing the package also
registers the equivalent methods on `pl.Expr.tokens`.

## `count()`

`count(expr, tokenizer=None, *, model=None, cache_capacity=None)` returns a
Polars `UInt32` expression with exact raw-text token counts. `expr` may be a
column name, `pl.Expr`, or `pl.Series`.

| Option | Meaning |
|---|---|
| `tokenizer` | One of `o200k_base`, `cl100k_base`, `p50k_base`, or `r50k_base`; defaults to `o200k_base`. |
| `model` | A versioned exact model alias; mutually exclusive with `tokenizer`. |
| `cache_capacity` | Optional FIFO cache size, 1–65,536 entries per worker task, for repeated String or Binary text. |

String, valid UTF-8 Binary, Categorical, and Enum inputs are accepted. Null
inputs produce null counts. Invalid non-null Binary UTF-8 raises a Polars
`ComputeError`; see [Binary input](binary-input.md). Input text is not
normalized, and special-token-looking substrings are ordinary raw text.

```python
import polars as pl
import polars_tokenizer as tokens

frame = pl.DataFrame({"text": ["hello world", "", None]})
counts = frame.select(tokens.count("text", tokenizer="o200k_base"))
```

The namespace form is `pl.col("text").tokens.count(...)`. It accepts the same
options except for the already-bound `expr` argument.

## `estimate_cost()`

`estimate_cost(expr, *, model, category="input",
usd_per_million_tokens=None, snapshot_date=None, cache_capacity=None)`
returns a `Float64` USD expression. It uses the exact local token count and a
pinned price snapshot; it is not a full API-request cost.

`category` is `input`, `cached_input`, or `output`. `snapshot_date` selects one
exact bundled snapshot date, not an inferred price interval. A non-negative,
finite `usd_per_million_tokens` overrides the bundled rate; it cannot be
combined with `snapshot_date`.

```python
costs = frame.select(
    tokens.estimate_cost("text", model="gpt-5", category="input").alias("cost_usd")
)
```

The namespace form is `pl.col("text").tokens.estimate_cost(...)`.

## `estimate_cost_details()`

`estimate_cost_details()` takes the same options as `estimate_cost()` and
returns a struct rather than a scalar. It includes:

| Field | Meaning |
|---|---|
| `token_count` | Exact `UInt32` count, null for null input. |
| `cost_usd` | `Float64` cost, null for null input. |
| `token_count_mode` | Currently `exact`. |
| `model`, `model_registry_version` | Selected alias and registry version. |
| `serving_provider`, `category`, `currency` | Pricing context. |
| `price_per_unit`, `unit_tokens` | Decimal-string USD rate and its integer token unit (currently 1,000,000). |
| `price_source`, `price_snapshot_date`, `price_registry_version`, `price_source_url` | Rate provenance. |

For a caller-supplied rate, `price_source` is `caller_override`; snapshot,
registry, provider, and source URL fields are null because they were not
verified by the bundled registry.

```python
details = frame.select(tokens.estimate_cost_details("text", model="gpt-5"))
```

The namespace form is `pl.col("text").tokens.estimate_cost_details(...)`.

## `price_info()`

`price_info(model, category="input", *, snapshot_date=None)` returns immutable
pricing metadata for an exact bundled snapshot. Its `price_per_unit` and
`price_per_token` values are `Decimal`, while `estimate_cost()` emits a Polars
`Float64` expression.

```python
rate = tokens.price_info("gpt-5", category="input")
print(rate.price_per_unit, rate.unit_tokens, rate.snapshot_date)
```

The bundled dates and model set are described in the
[provider capability matrix](provider-support.md). Use an explicit override
for private or negotiated rates. Neither the registry nor DataFrame
expressions make a runtime network call.
