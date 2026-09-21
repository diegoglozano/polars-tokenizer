# polars-tokenizer

`polars-tokenizer` is a native Polars expression plugin for exact, count-only
tokenization of string columns. The first milestone deliberately supports one
operation and one encoding:

```python
import polars as pl
import polars_tokenizer  # registers Expr.tokens

df = pl.DataFrame({"text": ["hello world", None, ""]})
out = df.with_columns(pl.col("text").tokens.count("o200k_base").alias("token_count"))
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
- Nulls are appended directly to a pre-sized `UInt32` output builder.
- The function is registered as elementwise, so Polars may schedule batches in
  its own runtime. The kernel does not create a nested thread pool.
- `o200k_base` vocabulary data is compiled into the wheel and pinned by
  `Cargo.lock`; execution performs no network access.
- Special-token-looking substrings are ordinary raw text. This matches
  `tiktoken.encode(text, disallowed_special=())`, not request/chat accounting.

The plugin measures only the supplied raw UTF-8 text. It does not include chat
wrappers, roles, tools, images, provider request serialization, or pricing.

## Development

Prerequisites are Rust 1.87+ and `uv`. A C-compatible linker is also required.

```bash
uv sync --group dev
uv run maturin develop --release
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run ty check
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

Run an end-to-end Polars/Python comparison and write machine-readable results:

```bash
uv run python benchmarks/run.py \
  --rows 100000 --length short --cardinality 1.0 \
  --output benchmarks/results/latest.json
```

The runner reports input bytes, rows/s, MiB/s, tokenizer initialization, warm
plugin execution, the Python `tiktoken` scalar baseline, peak RSS, environment
metadata, and a deterministic dataset hash. Use `--help` for dataset controls.
Benchmark numbers are not checked into this repository because no measurements
have yet been made on a controlled host.

## Correctness contract

For every valid Python string `s`:

```text
polars_tokenizer.count(s, "o200k_base")
    == len(tiktoken.get_encoding("o200k_base").encode(
           s, disallowed_special=()
       ))
```

Tests cover nulls, empty strings, embedded NULs, CRLF, combining and zero-width
characters, emoji, CJK, Arabic, chunked input, lazy/streaming execution, and
property-generated Unicode strings. Inputs are never normalized.

## Scope and roadmap

This repository implements Phase 1 only. In particular, `estimate`, model
aliases, cache controls, categorical specialization, fused aggregations, and
DataFrame-level analytics are not exposed yet. The next changes should be
driven by profiles and benchmark data in this order:

1. profile the exact kernel and quantify Polars/FFI overhead;
2. improve byte-balanced execution without nested parallelism;
3. benchmark categorical and bounded repeated-value caches;
4. add a separately measured estimator;
5. add `cl100k_base` and model aliases without coupling provider names to the
   tokenizer engine.

See [docs/architecture.md](docs/architecture.md) for the design boundaries and
[docs/benchmarking.md](docs/benchmarking.md) for the benchmark protocol.

## License

MIT
