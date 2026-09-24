# Roadmap and API boundaries

The order below keeps performance work evidence-driven and prevents provider
catalog concerns from leaking into the exact tokenizer kernel.

See the [provider capability matrix](provider-support.md) for the current
per-provider status of exact counting, estimated counting, and model-based cost
estimation.

## Exact-counting performance

- Implemented tokenizer definitions: `o200k_base` and `cl100k_base`.
- Establish controlled-host throughput, scaling, CPU, and peak-memory
  baselines with the benchmark matrix.
- Profile pre-tokenization, BPE merging, output construction, and Polars
  integration independently.
- Implemented: categorical/enum inputs count each used dictionary value once
  with dense, sparse, and parallel physical-ID paths.
- Benchmark bounded whole-value caches at 0.01%, 0.1%, 1%, 10%, 50%, and 100%
  cardinality before choosing an automatic policy.
- Investigate exceptionally large individual rows without introducing nested
  thread pools or changing exact results.

## Token estimation

- Freeze a multilingual, multi-format training and held-out corpus.
- Measure one-pass UTF-8 features by accuracy contribution per CPU cost.
- Fit deterministic, dependency-free estimators independently for each
  tokenizer family.
- Publish error distributions by language, content, length, token count, and
  ASCII/non-ASCII class.

## Model aliases

A versioned model registry may map a provider-facing model identifier to a
tokenizer definition. The low-level tokenizer continues to accept tokenizer
identifiers only. Model aliases live in the Python-facing or a separate
higher-level Rust layer.

Conceptually:

```text
model identifier -> model metadata -> tokenizer identifier -> count kernel
```

## Gemini local tokenizers (investigation TODO)

Google's experimental Python `LocalTokenizer` currently gives us a promising
reference path, but not yet a blanket exactness guarantee for every Gemini
model. Its Gemma 3 path loads a hash-pinned SentencePiece model. Its
`count_tokens()` implementation then calls `encode()` and sums the lengths of
the returned token-ID lists, so a Polars-native count-only kernel may avoid
substantial allocation and Python/API overhead on large columns. Newer mapped
models use a separate Gemma 4 Hugging Face tokenizer path and must be evaluated
independently.

The pinned Gemma 3 artifact has now been inspected and is a `SentencePiece`
BPE model, not a Unigram model. See
[the prototype notes](gemma3-prototype.md) for its validated configuration and
the reproducible oracle workflow.

Before adding public Gemini support:

- [ ] Benchmark Google's local tokenizer as shipped, its underlying
  SentencePiece batch operation, and remote `countTokens` separately. Record
  initialization/download time, cold and warm throughput, peak RSS, and
  allocations for scalar and batch workloads.
- [ ] Treat `gemma3`/`gemma4` as versioned tokenizer definitions and Gemini
  names as model aliases. Do not put Gemini-specific branching in the tokenizer
  kernel.
- [x] Prototype a pure-Rust, per-value, count-only SentencePiece BPE kernel
  for the pinned Gemma 3 model without materializing token IDs. The prototype
  is isolated behind the `gemma3-prototype` Cargo feature and is not yet a
  public Polars expression.
- [x] Preserve the pinned model's identity normalization, user-defined-symbol
  matching, ASCII-space escaping, and byte-fallback behavior exactly; do not
  apply an independent Unicode normalization pass. A 20,013-string initial
  oracle (3,321,397 bytes, 1,663,237 tokens) matched Google's local
  implementation.
- [ ] Compare against the official local implementation on the complete
  multilingual/fuzz corpus and against remote `countTokens` for stable model
  IDs. Publish any raw-text versus request-accounting differences.
- [ ] Reuse the existing Arrow, categorical/enum, byte-balanced parallel, and
  bounded-cache paths only after single-value parity is proven.
- [ ] Audit tokenizer artifact licensing and distribution. Pin the artifact
  URL, SHA-256, algorithm/configuration, and model-alias registry version; use
  an explicit verified download/cache flow if bundling is not permitted.
- [ ] Evaluate Gemma 4 separately, including its tokenizer artifact, processor
  behavior, dependency footprint, and whether a pure-Rust compatible path is
  possible.
- [ ] Add `gemma3` or `gemma4` to `count()` only after exactness is demonstrated
  for a pinned definition. If remote behavior cannot be reproduced locally,
  keep that mapping experimental or offer estimation rather than label it
  exact.

The initial scope remains raw text. Multimodal inputs, roles, tools, response
schemas, and provider request serialization belong to future request-level
accounting even where Google's local helper accepts some of those structures.

Reference implementations and definitions:

- [Google Gen AI Python local tokenizer](https://github.com/googleapis/python-genai/blob/main/google/genai/local_tokenizer.py)
- [Google Gen AI tokenizer loader and pinned model mappings](https://github.com/googleapis/python-genai/blob/main/google/genai/_local_tokenizer_loader.py)

## Model-based cost estimation (TODO)

Cost estimation is a higher-level analytics feature and will accept a model,
not a tokenizer. Tokenizers do not have prices; models and billing categories
do.

A future API may look like:

```python
pl.col("text").tokens.estimate_cost(
    model="gpt-5",
    category="input",
)
```

Before exposing that API:

- [ ] Define input, cached-input, and output categories explicitly.
- [ ] Store price, currency, unit, provider, effective date, and registry
  version.
- [ ] Pin pricing snapshots so results cannot change silently.
- [ ] Allow an explicit caller-supplied pricing override for private or negotiated
  prices.
- [ ] Resolve the model to its tokenizer independently from its price metadata.
- [ ] Report the token-count mode (`exact` or `estimate`) alongside the result.
- [ ] Keep raw-text cost estimates distinct from full request costs, which may also
  include wrappers, tools, images, audio, or provider serialization.
- [ ] Test boundary dates and unknown/retired model behavior.
- [ ] Avoid runtime network access in DataFrame expressions.

The initial operation should estimate the cost of the provided raw text in one
declared billing category. Full request accounting remains a separate future
feature.

## Later DataFrame analytics

- Per-column and combined-row descriptions.
- Fused total/min/max/mean/histogram operations that avoid materializing a
  count column.
- Sampled total/distribution estimates with confidence intervals.
