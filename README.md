# polars-tokenizer

`polars-tokenizer` is a native Polars expression plugin for exact, count-only
tokenization of string, categorical, and enum columns. The current foundation
deliberately supports one operation and one encoding:

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
- Categorical and enum columns count each used dictionary value once, then map
  counts through their physical IDs without expanding rows to strings. Dense,
  sparse, and parallel paths bound overhead across different mappings.
- Nulls are appended directly to a pre-sized `UInt32` output builder.
- The function is registered as elementwise and uses Polars' own thread pool
  for byte-balanced work above 512 KiB. It stays sequential when the caller is
  already parallel, preventing nested oversubscription.
- `o200k_base` vocabulary data is compiled into the wheel and pinned by
  `Cargo.lock`; execution performs no network access.
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

This repository implements the exact-counting foundation only. In particular,
`estimate`, model aliases, cache controls, fused aggregations, cost estimation,
and DataFrame-level analytics are not exposed yet. The next changes should be
driven by profiles and benchmark data in this order:

1. establish controlled-host performance and memory baselines;
2. benchmark bounded whole-value caches for non-categorical strings;
3. optimize exceptionally large individual rows without oversubscription;
4. add a separately measured estimator;
5. add `cl100k_base` and model aliases without coupling provider names to the
   tokenizer engine;
6. evaluate a pure-Rust, count-only SentencePiece/Unigram path for Google's
   pinned Gemma tokenizer definitions, with Gemini names kept as model aliases;
7. add model-based raw-text cost estimation using versioned pricing snapshots.

Cost estimation will accept a **model**, not a tokenizer. A model registry will
resolve both its tokenizer and its input/cached-input/output prices in a
higher-level layer. The tokenizer kernel will remain provider-agnostic, and
pricing updates will never silently alter a pinned calculation. See
[docs/roadmap.md](docs/roadmap.md) for the proposed boundary.

See [docs/architecture.md](docs/architecture.md) for the design boundaries and
[docs/benchmarking.md](docs/benchmarking.md) for the benchmark protocol.

## License

MIT
