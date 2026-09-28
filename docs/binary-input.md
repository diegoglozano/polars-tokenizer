# Binary columns containing UTF-8 text

`count()`, `estimate_cost()`, and `estimate_cost_details()` accept Polars
`Binary` columns when each non-null value contains valid UTF-8. The bytes are
interpreted as the exact raw text they encode: no normalization, replacement
decoding, or special-token handling is applied. Counts match a `String` column
containing the same text.

```python
import polars as pl
import polars_tokenizer as tokens

frame = pl.DataFrame({"text": pl.Series([b"hello world", None, "你好".encode()], dtype=pl.Binary)})
counts = frame.select(tokens.count("text", tokenizer="o200k_base"))
```

Invalid UTF-8 in a non-null value raises a Polars `ComputeError`. Null values
produce null output without inspecting the bytes behind their validity mask.
This matters for sliced or modified Arrow arrays, where a null view can retain
bytes from a previous value. The implementation does not first cast the whole
column to `String`, because that cast can reject invalid bytes hidden under a
null.

```python
bad = pl.DataFrame({"text": pl.Series([b"\xff"], dtype=pl.Binary)})
bad.select(tokens.count("text"))  # raises ComputeError: invalid UTF-8
```

The native kernel borrows binary views and validates each non-null value before
counting it. It keeps the same `UInt32` output, null propagation, opt-in bounded
`cache_capacity`, and byte-balanced parallel scheduling as String input.
All-null and all-empty batches skip tokenization. This support is for UTF-8
text stored as Binary, not arbitrary binary data.
