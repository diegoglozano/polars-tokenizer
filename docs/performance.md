# Exploratory performance baseline

These results are directional measurements from a shared 4-vCPU x86_64 Linux
host, not release claims. They exist to make the optimization decision
auditable. The corpus contains 100,000 mixed short strings (English prose,
code, JSON, URLs, Spanish, CJK, and emoji), totaling 13,360,390 logical UTF-8
bytes. Every GigaToken result was checked row-by-row against the plugin output.

Versions: Polars 1.36.1, GigaToken 0.10.0, and the official `o200k_base` rank
file with SHA-256
`446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`.

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

Re-run the committed end-to-end harness with fixed thread counts:

```bash
POLARS_MAX_THREADS=1 uv run python benchmarks/run.py --rows 100000
POLARS_MAX_THREADS=2 uv run python benchmarks/run.py --rows 100000
POLARS_MAX_THREADS=4 uv run python benchmarks/run.py --rows 100000
```
