# Exploratory performance baseline

These results are directional measurements from a shared 4-vCPU x86_64 Linux
host, not release claims. They exist to make the optimization decision
auditable. The corpus contains 100,000 mixed short strings (English prose,
code, JSON, URLs, Spanish, CJK, and emoji), totaling 13,360,390 logical UTF-8
bytes. Every GigaToken result was checked row-by-row against the plugin output.

Versions: Polars 1.36.1, GigaToken 0.10.0, and the official `o200k_base` rank
file with SHA-256
`446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`.

## `cl100k_base` exact-kernel baseline

After adding `cl100k_base`, the count-only Criterion benchmark exercised both
compiled tokenizer definitions on the same repeated mixed-language sample.
These are single-string kernel measurements on the shared Intel N95 host, not
end-to-end Polars results.

| Logical size | `o200k_base` MiB/s | `cl100k_base` MiB/s |
|---:|---:|---:|
| 32 B | 125.5 | 125.2 |
| 256 B | 125.8 | 128.1 |
| 4 KiB | 127.0 | 129.1 |
| 128 KiB | 126.6 | 127.7 |

Criterion detected no statistically significant `o200k_base` regression
against the preceding baseline. The shared expression path resolves one
`CoreBpe` before processing rows, so supporting the second tokenizer does not
add tokenizer-name dispatch inside the row loop.

## Two-tokenizer Polars baseline

The end-to-end matrix now selects either exact tokenizer. A release build on
the shared Intel N95 host counted 100,000 unique mixed short strings totaling
12,797,394 UTF-8 bytes. Every output row matched Python `tiktoken` 0.12.0.
The five warm repetitions used the same dataset for both tokenizers and thread
counts (SHA-256
`00b9d6346ac6e0b7882f5ed50880b4790d49b78a13caa6628f46a44ea0086ea2`).

| Tokenizer | 1 thread MiB/s | 4 threads MiB/s |
|---|---:|---:|
| `o200k_base` | 176 | 457 |
| `cl100k_base` | 158 | 456 |

These are directional shared-host measurements, not controlled-host release
claims. Older matrix peak-RSS results include the Python reference check and
should not be attributed to the plugin alone. The current matrix records
separate process peaks. Reproduce the comparison with:

```bash
uv run --no-sync python -m benchmarks.matrix \
  --rows 100000 --lengths short --cardinalities 1.0 \
  --contents mixed --dtypes string \
  --tokenizers o200k_base,cl100k_base \
  --threads 1,4 --warm-repeats 5 \
```

## Google local Gemma 3 baseline

An isolated exploratory run measured Google Gen AI 2.11.0, SentencePiece
0.2.1, and protobuf 6.32.1 on the same shared 4-vCPU x86_64 host. The corpus
contained 10,000 unique mixed short strings totaling 1,277,394 UTF-8 bytes.
All four paths produced exactly 403,011 tokens.

| Implementation | Median MiB/s | Rows/s | Tokens/s | Peak RSS |
|---|---:|---:|---:|---:|
| Google `LocalTokenizer`, one call per row | 1.07 | 8,765 | 353,243 | 161 MiB |
| Google `LocalTokenizer`, aggregate batch call | 0.319 | 2,618 | 105,507 | 186 MiB |
| SentencePiece scalar, token IDs | 3.03 | 24,833 | 1,000,803 | 161 MiB |
| SentencePiece batch, token IDs | 8.49 | 69,687 | 2,808,471 | 178 MiB |

Direct SentencePiece batch tokenization was about 8.0x faster than calling
Google's wrapper once per row and 26.6x faster than the wrapper's aggregate
batch form on this case. The latter result is not a per-row API: it returns one
total for the entire input. Source inspection indicates that the wrapper first
converts and traverses SDK content objects, then calls SentencePiece `encode()`
and sums materialized token-ID list lengths. The benchmark does not yet measure
a Rust count-only implementation.

The cached tokenizer initialized in approximately 0.12 seconds. The first
uncached smoke run downloaded and verified a 4,689,074-byte artifact and spent
0.85 seconds in tokenizer initialization; import time was approximately
1.1 seconds on cached runs. Treat these as directional measurements because
network, filesystem cache, process scheduling, and shared-host load affect
them.

Reproduce the correctness comparison and timings with the command in
[benchmarking.md](benchmarking.md#geminigemma-3-local-baseline). For isolated
peak-memory comparisons, repeat it once per `--implementation` value.

### Pure-Rust Gemma 3 count-only prototype

The first Rust feasibility kernel was then measured on a deterministic
20,013-row oracle containing 3,321,397 UTF-8 bytes and 1,663,237 tokens. Every
row matched Google's pinned Gemma 3 tokenizer. This corpus is more
Unicode-heavy than the 10,000-row baseline above, so only results within this
table are directly comparable.

| Implementation | Median MiB/s | Rows/s | Tokens/s | Relative |
|---|---:|---:|---:|---:|
| Rust count-only prototype | 17.4 | 110,000 | 9,140,000 | 1.00x |
| SentencePiece batch, token IDs | 7.64 | 48,268 | 4,011,430 | 0.44x |
| SentencePiece scalar, token IDs | 2.50 | 15,780 | 1,311,424 | 0.14x |

The count-only path was 2.28x faster than direct batch tokenization and 6.97x
faster than scalar tokenization on this corpus. Model parsing, validation, and
index construction took about 94 ms. The benchmark remains single-threaded and
below Polars; Arrow integration, workload partitioning, and memory measurement
are the next gates. See [gemma3-prototype.md](gemma3-prototype.md) for the
reproducible oracle and benchmark commands.

## Before and after byte-balanced parallelism

| Implementation | Median MiB/s | Relative to Phase 1 |
|---|---:|---:|
| Phase 1 sequential Polars plugin | 163 | 1.00x |
| Byte-balanced Polars plugin, 4 threads | 414 | 2.54x |
| GigaToken list input, 1 thread, token IDs | 114 | 0.70x |
| GigaToken list input, 4 threads, token IDs | 278 | 1.70x |
| GigaToken borrowed buffer, 4 threads, token IDs | 473 | 2.90x |

The optimized plugin is about 49% faster than GigaToken's parallel Python-list
path on this workload and within 14% of its preferred whole-buffer API. The
comparison is necessarily asymmetric: the plugin returns only `UInt32` counts,
while GigaToken 0.10.0 has no count-only operation and materializes token IDs.

## Thread scaling

| Polars threads | Median MiB/s | Speedup | Parallel efficiency |
|---:|---:|---:|---:|
| 1 | 171 | 1.00x | 100% |
| 2 | 252 | 1.47x | 74% |
| 4 | 393 | 2.30x | 58% |

The separate comparison run measured 414 MiB/s at four threads; the spread is
normal for this shared host. A threshold sweep found neutral or negative gains
at 250–400 KiB and a positive crossover near 500 KiB, which is why the runtime
cutoff is 512 KiB.

### Dominant individual rows

An isolated matrix case with 100 tiny rows and one 4 or 16 MiB middle row
matched the Python `tiktoken` reference for both tokenizers. Before the
dominant-row guard, four-thread runs spent about one core of CPU on these
cases: one row cannot be divided across the worker pool. The dispatcher now
stays sequential when the text outside the largest row totals less than
512 KiB, avoiding parallel task and output-concatenation overhead without
changing token counts. On this shared host, 20 warm repetitions after the
guard gave these median times:

| Tokenizer | Largest row | 1 thread | 4 threads |
|---|---:|---:|---:|
| `o200k_base` | 4 MiB | 31.0 ms | 30.6 ms |
| `o200k_base` | 16 MiB | 123.1 ms | 125.5 ms |
| `cl100k_base` | 4 MiB | 30.8 ms | 30.3 ms |
| `cl100k_base` | 16 MiB | 123.8 ms | 123.4 ms |

The nearly equal times are directional; this host is shared, and the guard
does not make a single row parallel. With 100,000 additional short rows, the
remaining text exceeds the guard threshold and the parallel path remains
available. The ordinary 100,000-row short-string workload still measured
about 178 MiB/s at one thread and 352 MiB/s at four threads in a follow-up
run. Controlled-host profiling is still needed before any intra-row strategy.

## Cardinality sweep before whole-value caching

The benchmark-matrix milestone measured 100,000 mixed 128-byte strings with
the current exact kernel, five warm repetitions, and reference parity enabled.
These are directional shared-host results. They establish the baseline that a
future categorical or whole-value cache must beat.

| Cardinality | 1 thread MiB/s | 4 threads MiB/s |
|---:|---:|---:|
| 0.01% | 264 | 719 |
| 0.1% | 174 | 471 |
| 1% | 167 | 374 |
| 10% | 162 | 376 |
| 50% | 173 | 431 |
| 100% | 176 | 413 |

Extreme repetition already benefits from the tokenizer's bounded internal
piece cache. A dedicated dictionary fast path may still avoid repeated
pre-tokenization and row-level dispatch, but its comparison baseline at 0.01%
cardinality is 719 MiB/s rather than the 413 MiB/s high-cardinality result.
The non-monotonic middle values should not be over-interpreted on this shared
host.

### Experimental bounded whole-value cache

The standalone Rust [cache crossover benchmark](benchmarking.md#whole-value-cache-crossover-experiment)
compares each count-only kernel with 256-entry and 4,096-entry FIFO caches.
The cache is rebuilt per 100,000-row batch; inputs are shuffled 128-byte
strings, and cached outputs are checked against uncached outputs before timing.
The first shared-host sweep used 10 Criterion samples, 0.5-second warm-up, and
1-second target measurement per case. Speedups below are relative to uncached
counting for the same tokenizer and dataset; they are directional, not a
controlled-host result.

```bash
cargo bench --bench cache_crossover -- \
  --sample-size 10 --warm-up-time 0.5 --measurement-time 1 \
  --noplot --discard-baseline
```

| Cardinality | `o200k_base` 256 | `o200k_base` 4,096 | `cl100k_base` 256 | `cl100k_base` 4,096 |
|---:|---:|---:|---:|---:|
| 0.01% | 6.13x | 6.18x | 7.02x | 6.83x |
| 0.1% | 6.42x | 6.04x | 5.65x | 6.02x |
| 1% | 1.01x | 5.85x | 1.25x | 6.73x |
| 10% | 0.82x | 1.10x | 0.79x | 1.11x |
| 50% | 0.80x | 0.82x | 0.81x | 0.78x |
| 100% | 0.77x | 0.77x | 0.81x | 0.70x |

At 1% cardinality, the 256-entry cache hit on 25,131 of 100,000 rows;
the 4,096-entry cache held all 1,000 values and hit on 99,000 rows. At 10%,
the 4,096-entry cache hit on 37,746 rows, leaving only a small timing gain.
At 100%, neither cache hit. This experiment excludes Polars dispatch,
Arrow output construction, nulls, parallelism, and peak RSS. It does not
justify a runtime cache policy yet.

### Polars-output cache cross-check

The follow-up [isolated Polars-output experiment](benchmarking.md#whole-value-cache-crossover-experiment)
includes `StringChunked` iteration and `UInt32Chunked` output construction.
Each cache mode ran in a fresh process on the same 100,000-row, 128-byte
corpus, with five warm repetitions and per-row output-hash parity. These
shared-host speedups are relative to uncached counting within the same case:

| Cardinality | `o200k_base` 256 | `o200k_base` 4,096 | `cl100k_base` 256 | `cl100k_base` 4,096 |
|---:|---:|---:|---:|---:|
| 0.01% | 13.77x | 11.71x | 14.24x | 14.42x |
| 0.1% | 12.96x | 16.29x | 14.44x | 11.86x |
| 1% | 1.09x | 10.88x | 0.99x | 10.88x |
| 10% | 0.85x | 1.08x | 0.90x | 1.33x |
| 50% | 0.83x | 0.87x | 0.84x | 0.85x |
| 100% | 0.88x | 0.76x | 0.88x | 0.85x |

The 1% null smoke case also preserved output parity. Process peak RSS varied
by less than about 0.3 MiB between modes within these cases, comparable to
run-to-run noise; this does not establish the cache's memory cost. The
experiment still excludes Python expression dispatch and parallel chunking.
There is no automatic or public cache setting yet.

### Native component cross-check

The [isolated component experiment](benchmarking.md#component-overhead-experiment)
used 100,000 rows, including 99,000 distinct 128-byte values and 1,000 nulls.
Each mode ran in its own process for seven warm repetitions, with identical
dataset and per-row count hashes. Shared-host median wall times were:

| Tokenizer | Rust vector count | Polars input, vector output | Polars input, Arrow output | Arrow output only |
|---|---:|---:|---:|---:|
| `o200k_base` | 99.0 ms | 93.7 ms | 90.5 ms | 0.44 ms |
| `cl100k_base` | 94.0 ms | 94.1 ms | 92.5 ms | 0.49 ms |

For this workload, constructing the output from precomputed counts is small
relative to exact counting. Differences among the three counting modes are
within shared-host variation and do not establish an input or output layout
speedup. Times are not additive, and this does not separate tokenizer
pre-tokenization from BPE merging or include Python expression dispatch.

The follow-up length sweep retained the same 100,000 rows, 1% nulls, four
modes, and output-hash parity, with seven warm repeats for each of two
tokenizers. Across both tokenizers and all three counting modes, shared-host
median times were:

| Non-null row bytes | Counting range | Output-only range |
|---:|---:|---:|
| 24 | 7.5–9.0 ms | 0.34–0.45 ms |
| 128 | 91–104 ms | 0.34–0.35 ms |
| 2,048 | 1.57–1.65 s | 0.35–0.58 ms |

Output construction is more visible for tiny values, but counting remains the
larger cost here. The tiny generator uses ASCII padding while the longer
values include repeated multilingual text, so this is not a pure length
scaling curve. Shared-host variation also prevents ranking the three counting
modes from these measurements.

## Categorical fast path

The same 100,000-row workload was run with string and categorical inputs after
adding direct physical-ID dispatch. Seven warm repetitions were used. MiB/s is
normalized to the original logical UTF-8 bytes for both representations.

| Cardinality | Threads | String MiB/s | Categorical MiB/s | Speedup |
|---:|---:|---:|---:|---:|
| 0.01% | 1 | 253 | 3,106 | 12.3x |
| 0.01% | 4 | 627 | 2,897 | 4.6x |
| 0.1% | 1 | 161 | 2,671 | 16.6x |
| 0.1% | 4 | 455 | 2,722 | 6.0x |
| 1% | 1 | 174 | 1,753 | 10.1x |
| 1% | 4 | 393 | 1,464 | 3.7x |
| 10% | 1 | 157 | 545 | 3.5x |
| 10% | 4 | 441 | 926 | 2.1x |
| 50% | 1 | 173 | 175 | 1.0x |
| 50% | 4 | 447 | 264 | 0.6x |
| 100% | 1 | 173 | 105 | 0.6x |
| 100% | 4 | 431 | 205 | 0.5x |

The fast path is compelling through 10% cardinality and reaches about 3 GiB/s
for extreme repetition. Nearly unique categorical input is slower than raw
strings because it gains no tokenization reuse while paying mapping and gather
costs; it remains supported without expanding the column to strings. Users
should not cast nearly unique strings to categorical solely for token counting.

The categorical comparison used `--skip-reference` to keep the matrix focused
on plugin timing. Separate categorical, enum, wide-ID, arbitrary-Unicode, and
string tests enforce exact reference parity.

Reproduce the comparison with:

```bash
uv run --no-sync python -m benchmarks.matrix \
  --rows 100000 --lengths short \
  --cardinalities 0.0001,0.001,0.01,0.1,0.5,1.0 \
  --contents mixed --dtypes string,categorical \
  --threads 1,4 --warm-repeats 7 --skip-reference
```

Reproduce the sweep with:

```bash
uv run --no-sync python -m benchmarks.matrix \
  --rows 100000 --lengths short \
  --cardinalities 0.0001,0.001,0.01,0.1,0.5,1.0 \
  --contents mixed --threads 1,4 --warm-repeats 5
```

Re-run the committed end-to-end harness with fixed thread counts:

```bash
POLARS_MAX_THREADS=1 uv run --no-sync python -m benchmarks.run --rows 100000
POLARS_MAX_THREADS=2 uv run --no-sync python -m benchmarks.run --rows 100000
POLARS_MAX_THREADS=4 uv run --no-sync python -m benchmarks.run --rows 100000
```
