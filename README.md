# polars-tokenizer

`polars-tokenizer` is a native Polars expression plugin for exact, count-only
tokenization of string, categorical, and enum columns. It exposes exact counts
and model-based raw-text cost estimates. Exact counting currently supports the
`o200k_base` and `cl100k_base` tokenizer definitions:

```python
import polars as pl
import polars_tokenizer  # registers Expr.tokens

df = pl.DataFrame({"text": ["hello world", None, ""]})
out = df.with_columns(pl.col("text").tokens.count("o200k_base").alias("token_count"))
```

Provider model aliases are resolved outside the tokenizer kernel:

```python
out = df.with_columns(pl.col("text").tokens.count(model="gpt-5").alias("token_count"))
```

Aliases are exact, not prefix matches, and are pinned by
`polars_tokenizer.MODEL_REGISTRY_VERSION`. Pass a tokenizer when reproducibility
should not depend on a provider model name.

Raw-text cost estimation is available for `gpt-4.1`, `gpt-4o`, and `gpt-5`
with pinned direct OpenAI price snapshots:

```python
costs = df.with_columns(
    pl.col("text").tokens.estimate_cost(model="gpt-5", category="input").alias("estimated_cost_usd")
)
```

The current price metadata is inspectable with `price_info("gpt-5")` and is
pinned by `PRICE_REGISTRY_VERSION`. Cost expressions perform no network access.
They count only the supplied raw text and exclude request wrappers, tools,
images, audio, and other provider-side accounting.

To select the bundled price snapshot explicitly, pass
`snapshot_date=polars_tokenizer.PRICE_SNAPSHOT_DATE` to `price_info()`,
`estimate_cost()`, or `estimate_cost_details()`. The selector matches a snapshot
date exactly; dates without a bundled snapshot fail. It does not infer a price
for dates between snapshots or report whether a model is still served.
The current snapshot is 2026-09-25; the 2026-09-24 GPT-5 snapshot remains
selectable for reproducibility.

For a cost column that carries its count mode and price provenance, use the
structured form:

```python
details = df.select(pl.col("text").tokens.estimate_cost_details(model="gpt-5").alias("estimate"))
```

Each struct contains the `UInt32` `token_count` used for pricing, `cost_usd`,
`token_count_mode` (currently `"exact"`), the model and billing category, and
the pinned rate's provider, date, and registry version. `price_per_unit` is a
decimal string to preserve the published rate.
For caller-supplied rates, `price_source` is `"caller_override"` and unverified
snapshot and serving-provider fields are null. `estimate_cost()` remains the
compact `Float64` expression.

Private or negotiated rates can be supplied explicitly without confusing a
model with a tokenizer:

```python
costs = df.select(
    pl.col("text").tokens.estimate_cost(
        model="gpt-4",
        category="input",
        usd_per_million_tokens=3.50,
    )
)
```

The functional form is also available and is friendlier to static type
checkers:

```python
import polars_tokenizer as tokens

df.select(tokens.count(pl.col("text"), tokenizer="o200k_base"))
```

## Why this implementation

- The Rust kernel calls `CoreBpe::count`, a dedicated count-only BPE path. It
  never creates token IDs.
- Polars string values are visited as borrowed `&str` views. No `Vec<String>`
  or `Vec<&str>` is materialized.
- Categorical and enum columns count each used dictionary value once, then map
  counts through their physical IDs without expanding rows to strings. Dense,
  sparse, and parallel paths bound overhead across different mappings.
- Nulls are appended directly to a pre-sized `UInt32` output builder.
- The function is registered as elementwise and uses Polars' own thread pool
  for byte-balanced work above 512 KiB. It stays sequential when the caller is
  already parallel, preventing nested oversubscription.
- `o200k_base` and `cl100k_base` vocabulary data are compiled into the wheel
  and pinned by `Cargo.lock`; execution performs no network access.
- Special-token-looking substrings are ordinary raw text. This matches
  `tiktoken.encode(text, disallowed_special=())`, not request/chat accounting.

The plugin measures only the supplied raw UTF-8 text. It does not include chat
wrappers, roles, tools, images, provider request serialization, or pricing.

## Development

Prerequisites are Rust 1.87+ and `uv`. A C-compatible linker is also required.

```bash
uv sync --group dev --no-install-project
uv run --no-sync maturin develop --release
uv run --no-sync pytest
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

The Python and Rust Polars versions are intentionally coupled because native
expression plugins use Polars' plugin ABI. When upgrading Polars, update
`polars`, `pyo3-polars`, and the Python dependency together.

## Benchmarks

Run the Rust count-only kernel benchmark:

```bash
cargo bench --bench exact_count
```

Run one end-to-end Polars/Python comparison and write machine-readable results:

```bash
uv run --no-sync python -m benchmarks.run \
  --rows 100000 --length short --content mixed --cardinality 1.0 \
  --output benchmarks/results/latest.json
```

Run an isolated matrix across workload and thread-count axes:

```bash
uv run --no-sync python -m benchmarks.matrix \
  --rows 1000,100000 \
  --lengths tiny,short,medium \
  --cardinalities 0.01,0.1,1.0 \
  --contents mixed,code,cjk \
  --dtypes string,categorical \
  --tokenizers o200k_base,cl100k_base \
  --threads 1,2,4 \
  --output benchmarks/results/matrix.json \
  --csv benchmarks/results/matrix.csv \
  --parquet benchmarks/results/matrix.parquet
```

The runners report input bytes, rows/s, MiB/s, tokens/s, process CPU usage,
tokenizer initialization, cold and median warm execution, the Python
`tiktoken` scalar baseline, peak RSS, environment metadata, and deterministic
dataset hashes. Matrix cases run in separate processes because Polars fixes its
thread pool at process startup. Inputs estimated above 512 MiB are skipped by
default; use `--max-input-mib` deliberately on larger hosts.

An optional, isolated benchmark for Google's experimental local Gemma 3
tokenizer compares its Python wrapper with direct SentencePiece scalar and
batch paths. It does not add Google, SentencePiece, Transformers, or PyTorch to
the project dependencies. See [docs/benchmarking.md](docs/benchmarking.md) for
the versioned `uv` command and measurement boundaries.

## Correctness contract

For every valid Python string `s` and supported tokenizer `t`:

```text
polars_tokenizer.count(s, tokenizer=t)
    == len(tiktoken.get_encoding(t).encode(
           s, disallowed_special=()
       ))
```

Tests cover nulls, empty strings, embedded NULs, CRLF, combining and zero-width
characters, emoji, CJK, Arabic, chunked input, lazy/streaming execution, and
property-generated Unicode strings. Inputs are never normalized.

## Scope and roadmap

The public surface currently includes exact counting, versioned OpenAI model
aliases, pinned cost estimation for GPT-4.1, GPT-4o, and GPT-5 with structured
provenance, and
caller-supplied price overrides.
Token-count estimation, cache controls, fused aggregations, and DataFrame-level
analytics are not exposed yet. The next changes should be driven by profiles
and benchmark data in this order:

1. establish controlled-host performance and memory baselines;
2. benchmark bounded whole-value caches for non-categorical strings;
3. optimize exceptionally large individual rows without oversubscription;
4. add a separately measured estimator;
5. expand dated model-price snapshots without coupling provider names to the
   tokenizer engine;
6. move the pure-Rust Gemma 3 prototype toward public Polars integration after
   broader parity and artifact-licensing checks.

Cost estimation accepts a **model**, not a tokenizer. A higher-level registry
resolves its tokenizer independently from dated input/cached-input/output price
snapshots. The tokenizer kernel remains provider-agnostic, and pricing updates
never silently alter a pinned calculation. See
[docs/roadmap.md](docs/roadmap.md) for the boundary and remaining work.

See [docs/architecture.md](docs/architecture.md) for the design boundaries and
[docs/benchmarking.md](docs/benchmarking.md) for the benchmark protocol.
The [provider capability matrix](docs/provider-support.md) tracks exact,
estimated, and cost-estimation support without conflating models with
tokenizers or serving providers.
The experimental Gemma 3 kernel and its parity workflow are documented in
[docs/gemma3-prototype.md](docs/gemma3-prototype.md).

## License

MIT
