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

Before adding caching, run every workload with cache disabled, automatic, and
forced at 0.01%, 0.1%, 1%, 10%, 50%, and 100% cardinality. Before adding an
estimator, freeze a held-out corpus and report MAE, median absolute error, MAPE,
p50/p95/p99/max relative error, and bias by language, content type, length, true
count, and ASCII/non-ASCII class.
