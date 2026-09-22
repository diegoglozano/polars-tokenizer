# Benchmark protocol

Treat bytes per second as the primary kernel metric. Rows per second is useful
only alongside the string-size distribution.

For each meaningful change, run both the Criterion kernel benchmark and the
end-to-end runner. Cover at least:

- 1K, 100K, 1M, and—on suitable hardware—10M rows;
- tiny, short, medium, and long strings;
- 1%, 10%, 50%, and 100% cardinality;
- prose, code, JSON/logs, URLs, Latin languages, CJK, and emoji;
- cold first execution and warm steady state;
- `POLARS_MAX_THREADS=1,2,4,8,...` where hardware permits.

Store JSON results under a host-specific directory outside version control or
in benchmark artifacts. Always retain the dataset hash and environment block.
Compare peak RSS as well as elapsed time. Do not claim improvements from noisy
shared CI runners; CI is for correctness and build regressions.

The single-case runner provides content, null-rate, warm-repeat, cardinality,
and size controls:

```bash
uv run --no-sync python -m benchmarks.run \
  --rows 100000 --length short --content code \
  --cardinality 0.1 --null-rate 0.01 --warm-repeats 7
```

The matrix runner takes comma-separated axes and starts a fresh process per
case so `POLARS_MAX_THREADS` is applied before Polars initializes:

```bash
uv run --no-sync python -m benchmarks.matrix \
  --rows 1000,100000 \
  --lengths tiny,short,medium,long \
  --cardinalities 0.01,0.1,0.5,1.0 \
  --contents mixed,english,code,json,logs,urls,spanish,cjk,emoji \
  --dtypes string,categorical \
  --threads 1,2,4,8 \
  --output /tmp/polars-tokenizer/matrix.json \
  --csv /tmp/polars-tokenizer/matrix.csv \
  --parquet /tmp/polars-tokenizer/matrix.parquet
```

The Cartesian product above is intentionally large. Start with one varying
axis at a time, then run selected interactions. Cases whose estimated input is
over 512 MiB are skipped unless `--max-input-mib` is raised. `--skip-reference`
is available for profiling very large cases, but release correctness runs must
retain the reference comparison.

For native CPU profiles, use the same deterministic case parameters with a
sampling profiler that can resolve Rust symbols. Keep profiling builds and
compiler flags in the result notes. Separate these regions when interpreting a
profile:

1. Polars expression/plugin dispatch;
2. byte-balanced column partitioning;
3. tokenizer pre-tokenization;
4. BPE merging/counting;
5. output-array construction.

## Gemini/Gemma 3 local baseline

The optional Gemini benchmark deliberately does not add Google's tokenizer or
its heavier extras to the project environment. For the current Gemma 3 path,
run an isolated, explicitly versioned environment with the lightweight direct
dependencies:

```bash
uv run --no-sync \
  --with google-genai==2.11.0 \
  --with sentencepiece==0.2.1 \
  --with protobuf==6.32.1 \
  python -m benchmarks.gemini_local \
  --model gemini-2.5-flash \
  --rows 10000 --length short --content mixed \
  --output /tmp/polars-tokenizer/gemini-local.json
```

The runner separately reports Google's `LocalTokenizer` scalar and aggregate
batch paths plus direct SentencePiece scalar and batch-ID paths. It verifies
that all selected paths agree, records the resolved package versions and
hash-pinned tokenizer artifact, and reports import, initialization, cold, warm,
CPU, throughput, and process peak-RSS measurements.

Run one `--implementation` per process when comparing peak RSS; the default
`all` mode is intended for correctness and timing comparisons in one report.
This benchmark supports only SDK mappings backed by the pinned SentencePiece
artifact. Gemma 4/Hugging Face mappings require a separate environment and
benchmark because their dependencies and processor behavior differ.

Before adding caching, run every workload with cache disabled, automatic, and
forced at 0.01%, 0.1%, 1%, 10%, 50%, and 100% cardinality. Before adding an
estimator, freeze a held-out corpus and report MAE, median absolute error, MAPE,
p50/p95/p99/max relative error, and bias by language, content type, length, true
count, and ASCII/non-ASCII class.
