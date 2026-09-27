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

The matrix runner takes comma-separated axes and starts separate plugin and
reference processes per case. This applies `POLARS_MAX_THREADS` before Polars
initializes and gives each implementation its own process peak RSS:

```bash
uv run --no-sync python -m benchmarks.matrix \
  --rows 1000,100000 \
  --lengths tiny,short,medium,long \
  --cardinalities 0.01,0.1,0.5,1.0 \
  --contents mixed,english,code,json,logs,urls,spanish,cjk,emoji \
  --dtypes string,categorical \
  --tokenizers o200k_base,cl100k_base \
  --threads 1,2,4,8 \
  --output /tmp/polars-tokenizer/matrix.json \
  --csv /tmp/polars-tokenizer/matrix.csv \
  --parquet /tmp/polars-tokenizer/matrix.parquet
```

The Cartesian product above is intentionally large. Start with one varying
axis at a time, then run selected interactions. Cases whose estimated input is
over 512 MiB are skipped unless `--max-input-mib` is raised. `--skip-reference`
is available for profiling very large cases, but release correctness runs must
retain the reference comparison. The runner verifies both processes generated
the same dataset and per-row token counts before combining their reports.

The single-case runner accepts `--tokenizer`; the matrix accepts
`--tokenizers`. Both default to `o200k_base`, and each case records the selected
tokenizer alongside its dataset hash and reference comparison. A direct
`benchmarks.run` invocation defaults to `--implementation all` for convenience;
its single peak-RSS value covers both implementations. Use `--implementation
plugin` or `--implementation reference` in separate processes for a fair
memory comparison.

For exceptionally large individual rows, `--outlier-bytes` replaces one
non-null row with deterministic text of exactly that UTF-8 byte length. The
default position is `middle`; `first` and `last` are also available. The
matrix accepts comma-separated outlier sizes, including `0` for the ordinary
control workload:

```bash
uv run --no-sync python -m benchmarks.matrix \
  --rows 100 --lengths tiny --cardinalities 0.5 \
  --outlier-bytes 0,1048576,4194304,16777216 \
  --outlier-position middle --tokenizers o200k_base,cl100k_base \
  --threads 1,4 --warm-repeats 5 \
  --output /tmp/polars-tokenizer/oversized-rows.json
```

Each case records the actual maximum row length and outlier index. The input
size guard includes the outlier, and the separate-process reference comparison
still checks every output row. This diagnostic does not split an individual
string across threads.

For native CPU profiles, use the same deterministic case parameters with a
sampling profiler that can resolve Rust symbols. Keep profiling builds and
compiler flags in the result notes. Separate these regions when interpreting a
profile:

1. Polars expression/plugin dispatch;
2. byte-balanced column partitioning;
3. tokenizer pre-tokenization;
4. BPE merging/counting;
5. output-array construction.

The Criterion kernel harness also compares direct `CoreBpe::count()` with
`CoreBpe::encode()` on identical strings for both tokenizers. Cases contain
exactly 32 bytes, 256 bytes, 4 KiB, or 128 KiB, and Criterion records their
actual byte throughput:

```bash
cargo bench --bench exact_count -- core_count_vs_encode
```

Each pair checks count/ID-length parity before timing. Both operations use
the crate's internal piece cache, while `encode()` materializes IDs; the
comparison is the full public operation difference, not an allocation-only
measurement.

## Component overhead experiment

Run the isolated native component comparison on a controlled host:

```bash
uv run --no-sync python -m benchmarks.component_overhead \
  --lengths tiny short medium --repeats 7 \
  --output /tmp/polars-tokenizer/component-overhead.json
```

Each subprocess uses a deterministic 100,000-row, 1%-null, all-unique dataset
with 24-, 128-, or 2,048-byte non-null values. `kernel_vec` counts borrowed
Python-free Rust strings into a vector;
`column_vec` counts the same text through a Polars `StringChunked` iterator
into a vector; `column_arrow` counts into a `UInt32Chunked`; `output_only`
builds that Arrow output from precomputed counts. The driver checks dataset
and every-row output hashes before comparing elapsed times. It reports warm
medians and Linux process RSS. These modes isolate coarse integration costs,
not tokenizer pre-tokenization versus BPE, and their times are not additive.
Every process holds both the Rust string vector and Polars input column, plus
precomputed counts, before timing; RSS is therefore not per-component memory.
The output-only case has no input-throughput metric because it does not
tokenize text. These modes also exclude the Python expression dispatcher and
production parallel partitioning; use the end-to-end matrix alongside them.
The 24-byte values contain ASCII padding and a unique suffix; longer values
also contain the repeated multilingual sample. Treat the sweep as a length
and content workload comparison, not a controlled length-only experiment.

## Whole-value cache crossover experiment

Run the isolated Rust experiment before considering a string-column cache:

```bash
cargo bench --bench cache_crossover
```

It compares uncached count-only calls with 256-entry and 4,096-entry bounded
FIFO caches for both exact tokenizers. Each case has 100,000 shuffled 128-byte
values and a guaranteed cardinality of 0.01%, 0.1%, 1%, 10%, 50%, or 100%.
The benchmark recreates the cache and output vector on every iteration,
checks exact count parity before timing, and reports cache hits and maximum
entries. Keys borrow from the stable input strings, so the experiment does not
measure key-copying costs.

These are kernel-level results, not a Polars expression or an automatic-cache
policy. They exclude Arrow output construction, null handling, parallel
dispatch, and process peak RSS. Compare them with the end-to-end matrix and
repeat on a controlled host before making a runtime change.

The follow-up Polars-output experiment constructs a real `StringChunked`,
counts into a `UInt32Chunked`, and runs each uncached/256-entry/4,096-entry mode
in its own process:

```bash
uv run --no-sync python -m benchmarks.cache_polars \
  --tokenizers o200k_base,cl100k_base \
  --cardinalities 0.0001,0.001,0.01,0.1,0.5,1.0 \
  --repeats 5 --output /tmp/polars-tokenizer/cache-polars.json
```

Use `--null-rows 1000` for a 1% null workload. The driver requires identical
dataset and per-row count hashes across modes. It reports warm wall time and,
on Linux, process RSS before and after counting plus process high-water RSS.
RSS includes Python-free native setup and the input column; it is not an
allocation count. This remains a benchmark-only prototype: it does not include
the Python/Polars expression dispatcher or parallel chunking, and it does not
enable a cache in production.

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

`benchmarks.estimator_eval.score_predictions()` implements those summaries for
supplied exact counts and estimates, including each listed stratum. It buckets
UTF-8 lengths at 24, 128, and 2,048 bytes and exact token counts at 0, 8, 32,
and 128. Percentage errors exclude zero-token references (which remain in
absolute-error and bias metrics); percentile errors use nearest-rank. This is
evaluation plumbing only. No training corpus, held-out corpus, fitted estimator,
or accuracy claim is included yet.

Before freezing a candidate corpus, use
`benchmarks.estimator_corpus.validate_corpus()` to verify unique sample IDs,
both train and held-out partitions, and that neither related `family_id`
variants nor identical text cross the partition boundary. Its order-independent
SHA-256 covers each record's text, split, source, language, content type, and
IDs with length-prefixed fields. `source_id` records provenance but does not
verify licensing or representativeness. The validator accepts empty text for
zero-token tests; no dataset is bundled by this module.

### Pinned external prose candidate

`python -m benchmarks.common_voice_corpus /tmp/common-voice-corpus` downloads
only Mozilla Common Voice `sentence-collector.txt` files at revision
`3ae618d5b34381ab154bcac562ed81aacf93e42c`, verifies each file's SHA-256,
and writes `records.jsonl` plus `manifest.json`. The ten selected languages are
Arabic, German, English, Spanish, French, Hindi, Japanese, Russian, Swahili,
and Simplified Chinese. Each contributes 256 train and 64 held-out sentences,
selected by stable text hashes; identical text always receives the same split.
The frozen candidate manifest SHA-256 is
`6e26888b3876d5a49b3f1e5606383e0786d622c2cbaf1740d192210291ba641f`.
The command requires network access; tests use offline synthetic fixtures.

Mozilla labels the `server/data` collection [CC0 1.0](https://github.com/common-voice/common-voice/blob/3ae618d5b34381ab154bcac562ed81aacf93e42c/server/data/LICENSE),
and its [Sentence Collector description](https://github.com/common-voice/sentence-collector#how-does-it-work)
states that submitted sentences must be public domain. We deliberately exclude
other source files, including Europarl and Wikipedia-derived text, because the
repository [distinguishes their provenance](https://github.com/common-voice/common-voice/tree/3ae618d5b34381ab154bcac562ed81aacf93e42c#licensing-and-content-source).
The checked-in script and hashes preserve provenance; generated sentence text
is not bundled with this package.

This is a licensed *candidate*, not a representative corpus: it has only short
prose, no code, markup, JSON, long documents, empty strings, or model-specific
exact counts. Its hash-based holdout prevents exact-text leakage, but the
source files do not identify paraphrases or translations as related families.
Of the 3,200 selected records, 401 are tiny (at most 24 UTF-8 bytes), 2,431
short (25–128 bytes), and 368 medium (129–2,048 bytes); none are long. 1,032
are ASCII-only, and the maximum observed length is 732 UTF-8 bytes.
Broader format coverage and an audited held-out source remain prerequisites
for publishing an estimator accuracy claim.

For exploratory exact-count labels, run
`python -m benchmarks.estimator_labels /tmp/common-voice-corpus` in the
development environment. The helper requires `tiktoken==0.12.0`, validates
the corpus manifest, and emits sorted `exact-labels.jsonl` plus an oracle-
versioned `exact-labels-manifest.json`. For this candidate, the label manifest
SHA-256 is `f9557668c7594606552ebe97a48c2b091c250a2471938f97b3cfdfc7a759ac4d`;
total exact raw-text counts are 70,806 for `cl100k_base` and 44,877 for
`o200k_base`. A local parity check found zero mismatches against
`polars_tokenizer.count()` on all 3,200 selected records for each encoding.
These labels are an independent oracle input to later estimator experiments,
not an accuracy result.

### Pinned structured-text candidate

`python -m benchmarks.json_schema_corpus /tmp/multiformat-corpus --base-corpus /tmp/common-voice-corpus`
downloads and SHA-256-checks the
[JSON Schema Test Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite)
archive at revision `5b0ee1613e45fcc2bddac00e07c19cd49b00d8a8`. The
upstream [MIT license](https://github.com/json-schema-org/JSON-Schema-Test-Suite/blob/5b0ee1613e45fcc2bddac00e07c19cd49b00d8a8/LICENSE)
applies; generated text is kept local and not bundled. Only the 46 top-level
draft-2020-12 JSON test files are selected, one intact file per record and
split family. The file-level hash split assigns 35 to train and 11 to held out.
The command verifies the base corpus manifest and rejects combined identity
or exact-text leakage.

The combined candidate has 3,246 records (2,595 train, 651 held out), ten
languages, and two content types, with corpus SHA-256
`74c52ca913659968486f5aeb8233ac37e0f8a27767da1f8802b2492b15150c96`.
Its 46 JSON files are all 637–50,423 UTF-8 bytes: 11 medium and 35 long by
the benchmark buckets. Running `benchmarks.estimator_labels` on the combined
corpus yields label SHA-256
`db8692d9d1392399beb75285f07f0fd1f14be8638d272297428ee88bcf50fa56`
with 147,521 `cl100k_base` and 121,608 `o200k_base` tokens. The labels
have zero mismatches against the local count kernel across all 3,246 records.
The JSON records are conformance fixtures, not typical application JSON;
JSON is English-only and
just 1.4% of records. More diverse structured text and code are still needed.

### Pinned Python-code candidate

`python -m benchmarks.python_code_corpus /tmp/three-format-corpus --base-corpus /tmp/multiformat-corpus`
adds whole Python source files from [Flask](https://github.com/pallets/flask/tree/d73fa1cdcbd8b1465c151db8924ba58b1dd14e35/src/flask)
and [Black](https://github.com/psf/black/tree/8d5a2d9f49378d7abe2eb632df1601818de8c24e/src/black).
Both have permissive licenses: Flask uses
[BSD-3-Clause](https://github.com/pallets/flask/blob/d73fa1cdcbd8b1465c151db8924ba58b1dd14e35/LICENSE.txt)
and Black uses [MIT](https://github.com/psf/black/blob/8d5a2d9f49378d7abe2eb632df1601818de8c24e/LICENSE).
The script pins both archive SHA-256 digests, does not bundle their text, and
preserves a source path and revision for every record. Any redistribution of
generated source files must retain the upstream license notices.

All 24 Flask library files are train records, and all 25 Black library files
are held out. This source-level split is stronger than splitting related
files within either project. One selected file is empty; 40 of the 49 are
longer than 2,048 UTF-8 bytes. The combined corpus has 3,295 records
(2,619 train, 676 held out), three content types, and SHA-256
`b7b08bb6a588bff6de4ffc5beb42445bb59183f6c19b41758c9759310cc61742`.
Its exact-label SHA-256 is
`4aafc493fc421c08385e8429340a2d499ea60f85acb803e0048ea3966f0270fe`;
totals are 344,489 `cl100k_base` and 319,511 `o200k_base` tokens, with zero
local-kernel parity mismatches. Code is labeled `language="und"` because the
natural-language field is not a programming-language identifier. This remains
exploratory: it covers only two Python projects, and code is 1.5% of records.

### Coverage audit

`python -m benchmarks.estimator_coverage /tmp/three-format-corpus` verifies the
corpus manifest, regenerates pinned exact labels, and prints split-by-format
coverage: record counts, natural-language labels, ASCII count, UTF-8 length
buckets and ranges, plus median/P95/max exact-token counts for each encoding.
Missing split/format combinations appear with zero records and null summary
statistics rather than being omitted.
It ties its output to both the corpus and exact-label SHA-256 values. On the
three-format candidate, held-out prose has 640 records but no long rows;
held-out JSON has 11 files (8 long), and held-out Python has 25 files (18
long). These uneven strata are visible for evaluation planning, not evidence
that the candidate represents production workloads.
