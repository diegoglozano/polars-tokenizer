# Architecture

## Current execution path

```text
Polars String Series / StringView chunks
              |
              | borrowed &str values + validity
              v
    o200k pre-tokenizer and count-only BPE
              |
              | one checked u32 per non-null row
              v
       Polars UInt32 output buffer
```

The Python layer constructs an expression and passes a small serialized
tokenizer identifier. It never sees row values. Polars transports Series across
the plugin ABI, and the Rust function iterates borrowed string views directly.

The count-only tokenizer pre-tokenizes text and computes the number of surviving
BPE parts. It does not collect token IDs. The static vocabulary is initialized
once and then shared immutably, which makes execution deterministic and safe
across Polars worker threads.

## Parallelism boundary

The expression is elementwise and batch independent. Polars can split and
schedule it while preserving lazy and streaming execution. The kernel is
intentionally sequential within a batch today: starting Rayon work inside a
Polars worker can oversubscribe the host. Byte-balanced intra-batch partitioning
belongs in a later phase only after profiles establish that Polars scheduling is
insufficient for highly skewed rows.

## Version boundary

Tokenizer and plugin ABI versions are pinned in `Cargo.lock` and `uv.lock`.
Only tokenizer identifiers enter the Rust engine. Future provider model aliases
must be resolved in a separate registry before dispatch and must never change a
tokenizer definition in place.

## Memory bound

Aside from one-time immutable vocabulary state, a call retains input buffers,
one UInt32 value per row, a validity bitmap when nulls exist, and bounded
tokenizer scratch state. It does not retain input strings or allocate token ID
arrays. The output builder deliberately reports an error rather than truncate if
an individual count exceeds `UInt32::MAX`.

## Deferred paths

- Dictionary/categorical dispatch needs an ABI-correct path that counts each
  dictionary value once and gathers through physical keys.
- Whole-value caching needs measured cardinality crossover points and a bounded,
  low-contention design.
- Estimation must use a distinct kernel and publish corpus-stratified error
  metrics. It must never be reachable through `count`.
- Fused scalar aggregations need separate plugin functions because expression
  composition alone necessarily materializes a count Series.
