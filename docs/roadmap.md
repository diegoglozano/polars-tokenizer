# Roadmap and API boundaries

The order below keeps performance work evidence-driven and prevents provider
catalog concerns from leaking into the exact tokenizer kernel.

## Exact-counting performance

- Establish controlled-host throughput, scaling, CPU, and peak-memory
  baselines with the benchmark matrix.
- Profile pre-tokenization, BPE merging, output construction, and Polars
  integration independently.
- Benchmark a categorical/dictionary fast path that counts each dictionary
  value once.
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
