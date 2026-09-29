# Exact token counts in Polars

`polars-tokenizer` adds native, count-only token expressions to Polars. It
counts raw UTF-8 text without building token-ID arrays and returns a `UInt32`
column that can be used in eager, lazy, and streaming queries.

```python
import polars as pl
import polars_tokenizer as tokens

frame = pl.DataFrame({"text": ["hello world", None, ""]})
counts = frame.select(tokens.count("text").alias("token_count"))
assert counts["token_count"].to_list() == [2, None, 0]
```

The default tokenizer is `o200k_base`. You can instead choose
`cl100k_base`, `p50k_base`, or `r50k_base`, or pass a supported model alias.
String, UTF-8 Binary, Categorical, and Enum columns are supported. Nulls stay
null, and special-token-looking text is counted as ordinary text.

Start with the [getting-started guide](getting-started.md), then see the
[API reference](api.md) for count, cost, cache, and pricing options. The
[Binary input guide](binary-input.md) explains UTF-8 validation and nulls.

For implementation and reproducibility details, see the
[architecture](architecture.md), [benchmarking protocol](benchmarking.md),
and [provider capability matrix](provider-support.md).

The library counts only the supplied raw text. It does not estimate a complete
provider request, including roles, wrappers, tools, images, or audio.
