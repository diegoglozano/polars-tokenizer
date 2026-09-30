# polars-tokenizer

Read the [documentation](https://diegoglozano.github.io/polars-tokenizer/)
for installation, examples, API options, and implementation notes.

Install from PyPI with `python -m pip install polars-tokenizer`.
See the [source installation steps](docs/getting-started.md#install-from-this-repository)
if you are developing the package or need to build for an unsupported platform.

`polars-tokenizer` is a native Polars expression plugin for exact, count-only
tokenization of String, UTF-8 Binary, Categorical, and Enum columns. It exposes
exact counts and model-based raw-text cost estimates. Exact counting supports
`o200k_base`, `cl100k_base`, `p50k_base`, and `r50k_base` tokenizer definitions:

```python
import polars as pl
import polars_tokenizer  # registers Expr.tokens

df = pl.DataFrame({"text": ["hello world", None, ""]})
out = df.with_columns(pl.col("text").tokens.count("o200k_base").alias("token_count"))
```

## Measured performance

On a four-core Intel N95, counting 100,000 distinct, roughly 128-byte String
values took **27.4 ms** after the first call: **3.65 million rows/s** or
**445 MiB/s** of input text (**132 million counted tokens/s**). Results for
other input shapes:

| Input | First call | Warm median | Rows/s | Input MiB/s | Counted tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 String rows, ~24 bytes each | 58.5 ms | 0.282 ms | 3.55 million | 80.9 | 28.9 million |
| 100,000 String rows, ~24 bytes each | 73.4 ms | 6.72 ms | 14.9 million | 338 | 129 million |
| 100,000 String rows, ~128 bytes each | 84.6 ms | 27.4 ms | 3.65 million | 445 | 132 million |
| 10,000 String rows, ~2 KiB each | 99.9 ms | 39.8 ms | 251,000 | 490 | 138 million |
| 100,000 Categorical rows, ~128 bytes each, 1% distinct | 132 ms | 6.54 ms | 15.3 million | 1,862 | 544 million |

These are end-to-end `DataFrame.select()` timings for the release build at
commit `943d8b3`, measured on 2026-09-29 with `o200k_base`, four Polars
threads, Python 3.14.7, and Polars 1.36.1. The input frame was already built;
the first call includes tokenizer initialization, while the warm number is the
median of nine later calls. The deterministic synthetic mixed-text inputs have
no nulls and are distinct except for the stated Categorical case. MiB/s uses
logical UTF-8 input bytes, including repeats. Tokens/s counts output tokens
for every row. On the 100,000-row short String case, a separate Python loop
using `tiktoken` 0.12.0 and calling `encode()` once per row took 2.50 s. That
is a scalar Python baseline, not a batch API comparison. Timings vary with
text, hardware, and thread count. The
[raw benchmark report](benchmarks/published/2026-09-29-intel-n95.json) records
every sample, dataset hash, reference count check, and reproduction command.

A 2026-09-30 follow-up on the same host measured 100,000 identical ~128-byte
String values at **1.53 ms** warm (**65.2 million rows/s**, **7,963 MiB/s** of
logical input), down from 14.45 ms before the homogeneous-value fast path.
With 1% nulls, the warm median fell from 13.22 to **1.68 ms**. The distinct-value
control stayed near 29 ms. These are medians of 15 calls; the [before/after
report](benchmarks/published/2026-09-30-uniform-string.json) contains the
samples and reference count checks.

Provider model aliases are resolved outside the tokenizer kernel:

```python
out = df.with_columns(pl.col("text").tokens.count(model="gpt-5").alias("token_count"))
```

Aliases are exact, not prefix matches, and are pinned by
`polars_tokenizer.MODEL_REGISTRY_VERSION`. Pass a tokenizer when reproducibility
should not depend on a provider model name.
The legacy `p50k_base` and `r50k_base` encodings are available by explicit
name; this does not add legacy model or pricing aliases. Their distinct
definitions are described in the [OpenAI tokenizer guide](https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken).

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

Binary columns containing UTF-8 text are accepted without a full-column
string conversion. Invalid UTF-8 in a non-null value raises an error; null
values stay null even if their underlying bytes are invalid. See
[Binary input](docs/binary-input.md) for details.

```python
import polars as pl
import polars_tokenizer as tokens

binary_df = pl.DataFrame(
    {"text": pl.Series([b"hello world", None, "你好".encode()], dtype=pl.Binary)}
)
binary_counts = binary_df.select(tokens.count("text"))
```

For String or Binary columns with many repeated values, opt into a bounded
whole-value cache:

```python
df.select(tokens.count("text", cache_capacity=4096))
```

The cache stores borrowed text and exact counts, with FIFO eviction and a
limit of 1–65,536 entries per worker task. It does not change results or apply
to categorical/enum columns, which already deduplicate values. Leave it off
for mostly unique strings: hashing and eviction can reduce throughput.
The same `cache_capacity` option is available on `estimate_cost()` and
`estimate_cost_details()`, including their `Expr.tokens` forms.

## Why this implementation

- The Rust kernel calls `CoreBpe::count`, a dedicated count-only BPE path. It
  never creates token IDs.
- Polars String values are visited as borrowed `&str` views; Binary values are
  validated as UTF-8 only when non-null. No full-column `Vec<String>` or
  `Vec<&str>` is materialized.
- An optional bounded cache reuses counts for repeated ordinary text
  without copying string data; the default path has no cache overhead.
- Large String columns with one repeated non-null value count it once and
  build constant output. Columns with few nulls reuse their input validity
  bitmap.
- Categorical and enum columns count each used dictionary value once, then map
  counts through their physical IDs without expanding rows to strings. Dense,
  sparse, and parallel paths bound overhead across different mappings.
- Nulls are appended directly to a pre-sized `UInt32` output builder.
- All-empty text and all-null String, Binary, Categorical, or Enum batches
  produce constant output directly, without loading a vocabulary or doing
  per-row tokenizer work. Empty values inside mixed batches also return zero
  before entering the tokenizer or optional cache.
- The function is registered as elementwise and uses Polars' own thread pool
  for byte-balanced work above 512 KiB. It stays sequential when the caller is
  already parallel, preventing nested oversubscription.
- All four supported vocabulary definitions are compiled into the wheel
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

Preview the MkDocs site locally:

```bash
uv run --group docs mkdocs serve
```

The Python and Rust Polars versions are intentionally coupled because native
expression plugins use Polars' plugin ABI. When upgrading Polars, update
`polars`, `pyo3-polars`, and the Python dependency together.

Maintainers: see the [release guide](docs/releasing.md) for wheel validation,
trusted publishing, and versioning.

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
provenance, caller-supplied price overrides, and opt-in bounded caching.
Token-count estimation, fused aggregations, and DataFrame-level analytics are
not exposed yet. The next changes should be driven by profiles
and benchmark data in this order:

1. establish controlled-host performance and memory baselines;
2. establish a controlled-host cache crossover and peak-memory profile;
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
