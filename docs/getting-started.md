# Getting started

## Install from this repository

The project is currently built as a native Python extension. From a clone,
install Rust 1.87+ and `uv`, then run:

```bash
uv sync --group dev --no-install-project
uv run --no-sync maturin develop --release
```

The Python and Rust Polars dependencies are pinned together because native
expression plugins use Polars' plugin ABI.

## Count text

Importing `polars_tokenizer` also registers the `Expr.tokens` namespace. The
functional API works with a column name or Polars expression:

```python
import polars as pl
import polars_tokenizer as tokens

frame = pl.DataFrame({"text": ["hello world", None, ""]})
counts = frame.select(tokens.count("text", tokenizer="o200k_base"))
assert counts["text"].to_list() == [2, None, 0]
```

The namespace API composes naturally with lazy and streaming plans:

```python
result = (
    frame.lazy()
    .with_columns(pl.col("text").tokens.count(model="gpt-5").alias("token_count"))
    .collect(engine="streaming")
)
```

Use either `tokenizer` or `model`, not both. The default tokenizer is
`o200k_base`. Model aliases are exact, versioned mappings rather than prefix
matches; see [provider support](provider-support.md).

## Count repeated values

Categorical and Enum columns automatically count each used dictionary value
once. For String or Binary columns with many repeated values, an optional
bounded FIFO cache can improve throughput:

```python
cached = frame.select(tokens.count("text", cache_capacity=4096))
```

The capacity is 1–65,536 entries per worker task. Leave caching off for mostly
unique text, where hashing and eviction can be slower.

## Estimate raw-text cost

For a supported model, cost expressions multiply exact local counts by a
pinned USD-per-million-token price. The category is `input` by default; the
other choices are `cached_input` and `output`.

```python
costs = frame.select(tokens.estimate_cost("text", model="gpt-5", category="input"))
details = frame.select(tokens.estimate_cost_details("text", model="gpt-5"))
```

`estimate_cost()` returns a `Float64` USD expression. The structured form also
includes the exact token count, count mode, price, snapshot date, registry
version, and source. Use [`price_info()`](api.md#price_info) to inspect the
bundled rate before building an expression.

These are **raw-text** estimates, not provider invoices or request-level
accounting. They exclude messages, roles, tools, images, audio, and other
provider serialization.
