# Gemma 3 count-only prototype

The `gemma3-prototype` Cargo feature contains an experimental, pure-Rust
`SentencePiece` BPE count kernel. It is deliberately not connected to the
Python or Polars API yet.

The kernel accepts model bytes rather than a Gemini model name. It validates
and implements the configuration in Google's pinned Gemma 3 artifact:

- BPE model type;
- identity normalization with no dummy prefix or whitespace collapsing;
- ASCII-space escaping to `▁`;
- longest-prefix matching and freezing of user-defined symbols;
- score-priority BPE merges with deterministic tie-breaking; and
- one token per UTF-8 byte for unknown final symbols.

It rejects unsupported configurations, including `UNUSED` pieces, rather than
returning an approximate result. It stores no token IDs while counting.
Vocabulary lookup uses a fast non-cryptographic hash because the fixed keys
come only from the verified model artifact; this trust boundary must remain
explicit if artifact loading is exposed publicly.

Google currently pins this artifact at SHA-256
`1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c`.
The prototype does not download or redistribute it. Artifact acquisition,
license review, hash verification, tokenizer versioning, and Gemini model alias
resolution remain higher-layer work.

## Reproduce the reference oracle

Generate counts using Google's explicitly versioned local dependencies:

```bash
uv run --no-sync \
  --with google-genai==2.11.0 \
  --with sentencepiece==0.2.1 \
  --with protobuf==6.32.1 \
  python -m benchmarks.gemma3_oracle \
  --output /tmp/gemma3-reference-oracle.jsonl
```

The Google loader prints the cached artifact path in the generated metadata.
Verify its SHA-256, then run the ignored Rust parity test:

```bash
sha256sum /tmp/vertexai_tokenizer_model/<cache-key>

POLARS_TOKENIZER_GEMMA3_MODEL=/tmp/vertexai_tokenizer_model/<cache-key> \
POLARS_TOKENIZER_GEMMA3_ORACLE=/tmp/gemma3-reference-oracle.jsonl \
cargo test --features gemma3-prototype \
  sentencepiece_bpe::tests::matches_gemma3_reference_oracle \
  -- --ignored
```

The initial default corpus contains 20,013 deterministic mixed, Unicode-fuzz,
long, normalization, embedded-NUL, and user-defined-symbol cases. This is an
initial feasibility gate, not yet the complete correctness corpus required for
public exact support.

## Benchmark the native kernel

Model initialization and steady-state counting are separate benchmark groups.
Setting the oracle path also adds a replay of the complete generated corpus:

```bash
POLARS_TOKENIZER_GEMMA3_MODEL=/tmp/vertexai_tokenizer_model/<cache-key> \
POLARS_TOKENIZER_GEMMA3_ORACLE=/tmp/gemma3-reference-oracle.jsonl \
cargo bench --features gemma3-prototype --bench gemma3_count
```

This benchmark is single-threaded and below the Arrow/Polars layer. It is meant
to establish the kernel baseline before parallel chunk execution and public API
integration are attempted.

On the exploratory 4-core Intel N95 host, the optimized prototype initialized
in about 94 ms. Its final median throughput was 10.4 MiB/s for the tiny-string
case, 10.0 MiB/s around 256 bytes, 6.4 MiB/s around 4 KiB, and 2.95 MiB/s around
128 KiB. The Unicode-heavy reference oracle reached 17.4 MiB/s because much of
that corpus takes the cheaper byte-fallback path.

On that identical 20,013-row oracle, direct SentencePiece 0.2.1 reached
7.64 MiB/s in batch mode and 2.50 MiB/s in a scalar loop while materializing
token IDs. The Rust count-only prototype was therefore 2.28x and 6.97x faster,
respectively. These shared-host, single-thread measurements are directional,
not release claims.
