# Exploratory performance baseline

These results are directional measurements from a shared 4-vCPU x86_64 Linux
host, not release claims. They exist to make the optimization decision
auditable. The corpus contains 100,000 mixed short strings (English prose,
code, JSON, URLs, Spanish, CJK, and emoji), totaling 13,360,390 logical UTF-8
bytes. Every GigaToken result was checked row-by-row against the plugin output.

Versions: Polars 1.36.1, GigaToken 0.10.0, and the official `o200k_base` rank
file with SHA-256
`446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`.

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
