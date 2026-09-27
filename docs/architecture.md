# Architecture

## Current execution path

```text
Polars String Series              Categorical / Enum Series
   | borrowed StringViews          | physical IDs + mapping
   v                               v
byte-balanced count kernel    count each used value once
   |                               | dense/sparse lookup
   +---------------+---------------+
                   |
                   v
         Polars UInt32 output buffer
```

The Python layer constructs an expression and passes a small serialized
tokenizer identifier and optional cache capacity. It never sees row values.
Polars transports Series across the plugin ABI, and the Rust function iterates
borrowed string views directly.

The count-only tokenizer pre-tokenizes text and computes the number of surviving
BPE parts. It does not collect token IDs. The static vocabulary is initialized
once and then shared immutably, which makes execution deterministic and safe
across Polars worker threads.

## Parallelism boundary

The expression is elementwise and batch independent. For calls of at least 512
KiB where Polars reports that the caller is not already parallel, the kernel
partitions contiguous rows by logical UTF-8 bytes and executes four tasks per
worker on Polars' own thread pool. Collecting the indexed parallel iterator
preserves row order. Smaller calls and already-parallel contexts stay
sequential, avoiding dispatch overhead and nested oversubscription.

Categorical and enum inputs retain their physical u8/u16/u32 IDs. Small or
low-cardinality mappings use a one-pass dense lookup. A sparse hash lookup
prevents a large shared category registry from forcing memory proportional to
the registry. When unique categorical text exceeds the same 512 KiB threshold,
its distinct values are counted on Polars' pool before a sequential ID gather.
Unused enum categories are never tokenized.

Rows are indivisible. One exceptionally large string can therefore dominate a
partition; splitting safely at tokenizer pre-token boundaries is a separate
future optimization.

For repeated ordinary strings, callers can opt into a FIFO cache of exact
counts. Its keys borrow input string views, and each sequential or parallel
task has its own cache with at most the requested number of entries. The
default path allocates no cache. Categorical and enum inputs already deduplicate
their values and do not use this cache.

## Version boundary

Tokenizer and plugin ABI versions are pinned in `Cargo.lock` and `uv.lock`.
Only tokenizer identifiers and execution settings enter the Rust engine.
Provider model aliases are resolved in a separate registry before dispatch and
must never change a tokenizer definition in place.

## Memory bound

Aside from one-time immutable vocabulary state, a call retains input buffers,
one UInt32 value per row, a validity bitmap when nulls exist, and bounded
tokenizer scratch state. Opt-in string caching adds at most the requested
number of borrowed keys and counts per task, plus FIFO bookkeeping.
Categorical inputs additionally use a count lookup
proportional to their mapping when dense, or to encountered values when the
mapping is disproportionately large. The kernel does not copy input strings
or allocate token ID arrays. The output builder deliberately reports an error
rather than truncate if an individual count exceeds `UInt32::MAX`.

## Deferred paths

- Automatic whole-value caching needs controlled-host crossover and peak-memory
  measurements; only explicit opt-in caching is available now.
- Estimation must use a distinct kernel and publish corpus-stratified error
  metrics. It must never be reachable through `count`.
- Fused scalar aggregations need separate plugin functions because expression
  composition alone necessarily materializes a count Series.
