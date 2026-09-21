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

Before adding caching, run every workload with cache disabled, automatic, and
forced at 0.01%, 0.1%, 1%, 10%, 50%, and 100% cardinality. Before adding an
estimator, freeze a held-out corpus and report MAE, median absolute error, MAPE,
p50/p95/p99/max relative error, and bias by language, content type, length, true
count, and ASCII/non-ASCII class.
