//! Native count-only token expressions for Polars.

use polars::prelude::*;
use pyo3_polars::PolarsAllocator;
use pyo3_polars::derive::{CallerContext, polars_expr};
use pyo3_polars::export::polars_arrow::array::ValueSize;
use pyo3_polars::export::polars_core::{POOL, THREAD_POOL};
use rayon::prelude::*;
use serde::Deserialize;

#[global_allocator]
static ALLOCATOR: PolarsAllocator = PolarsAllocator::new();

const O200K_BASE: &str = "o200k_base";
const PARALLEL_MIN_BYTES: usize = 512 * 1024;
const TASKS_PER_THREAD: usize = 4;

#[derive(Deserialize)]
struct CountKwargs {
    tokenizer: String,
}

/// Count `o200k_base` tokens without materializing token IDs.
///
/// Special-token-looking byte sequences are treated as ordinary text. The
/// vocabulary is initialized once by the `tiktoken` crate and then shared.
///
/// # Panics
///
/// Panics only when the binary was built without the required
/// `vocab-o200k_base` Cargo feature, which is an internal build invariant.
#[inline]
#[must_use]
pub fn count_o200k(text: &str) -> usize {
    tiktoken::get_encoding(O200K_BASE)
        .expect("o200k_base vocabulary must be compiled in")
        .count(text)
}

fn count_chunk(strings: &StringChunked) -> PolarsResult<UInt32Chunked> {
    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(strings.name().clone(), strings.len());

    // Iteration borrows values from Polars' StringView buffers. The only
    // per-chunk allocation here is the final values/validity output.
    for value in strings {
        match value {
            None => output.append_null(),
            Some(text) => {
                let count = u32::try_from(count_o200k(text)).map_err(|_| {
                    PolarsError::ComputeError("token count exceeds the UInt32 output range".into())
                })?;
                output.append_value(count);
            }
        }
    }

    Ok(output.finish())
}

/// Split contiguous rows into approximately byte-balanced ranges.
///
/// A row is never split, so one exceptionally large value may still dominate
/// a range. Empty and null rows remain attached to a neighboring range.
fn byte_balanced_ranges(
    strings: &StringChunked,
    total_bytes: usize,
    max_parts: usize,
) -> Vec<(usize, usize)> {
    debug_assert!(max_parts > 0);
    if strings.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::with_capacity(max_parts.min(strings.len()));
    let mut start = 0;
    let mut range_bytes = 0;
    let mut remaining_bytes = total_bytes;
    let mut remaining_parts = max_parts.min(strings.len());

    for (index, value) in strings.into_iter().enumerate() {
        range_bytes += value.map_or(0, str::len);
        let target = remaining_bytes.div_ceil(remaining_parts);
        let rows_remaining = strings.len() - index - 1;

        if remaining_parts > 1 && range_bytes >= target && rows_remaining > 0 {
            ranges.push((start, index + 1 - start));
            start = index + 1;
            remaining_bytes = remaining_bytes.saturating_sub(range_bytes);
            remaining_parts -= 1;
            range_bytes = 0;
        }
    }

    if start < strings.len() {
        ranges.push((start, strings.len() - start));
    }
    ranges
}

fn count_parallel(strings: &StringChunked, total_bytes: usize) -> PolarsResult<UInt32Chunked> {
    let n_threads = THREAD_POOL.current_num_threads();
    let max_parts = n_threads.saturating_mul(TASKS_PER_THREAD).max(1);
    let ranges = byte_balanced_ranges(strings, total_bytes, max_parts);

    let chunks = POOL.install(|| {
        ranges
            .into_par_iter()
            .map(|(offset, len)| {
                let offset = i64::try_from(offset).map_err(|_| {
                    PolarsError::ComputeError("row offset exceeds the Int64 range".into())
                })?;
                count_chunk(&strings.slice(offset, len))
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;
    let arrays = chunks
        .into_iter()
        .flat_map(|chunk| chunk.downcast_iter().cloned().collect::<Vec<_>>());
    Ok(UInt32Chunked::from_chunk_iter(
        strings.name().clone(),
        arrays,
    ))
}

#[polars_expr(output_type=UInt32)]
#[allow(clippy::needless_pass_by_value)] // Required by the plugin macro ABI.
fn token_count(
    inputs: &[Series],
    context: CallerContext,
    kwargs: CountKwargs,
) -> PolarsResult<Series> {
    if kwargs.tokenizer != O200K_BASE {
        return Err(PolarsError::ComputeError(
            format!(
                "unsupported tokenizer {:?}; supported tokenizers: {O200K_BASE}",
                kwargs.tokenizer
            )
            .into(),
        ));
    }

    let strings = inputs[0].str()?;
    let total_bytes = strings.get_values_size();
    let should_parallelize = !context.parallel()
        && THREAD_POOL.current_num_threads() > 1
        && total_bytes >= PARALLEL_MIN_BYTES;
    let output = if should_parallelize {
        count_parallel(strings, total_bytes)?
    } else {
        count_chunk(strings)?
    };

    Ok(output.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_o200k_counts() {
        let cases = [
            ("", 0),
            ("hello world", 2),
            ("hello\0world", 3),
            ("你好，世界", 3),
            ("👋🏽🌍", 6),
            ("e\u{301}", 2),
            ("line one\r\nline two", 5),
        ];

        for (text, expected) in cases {
            assert_eq!(count_o200k(text), expected, "input: {text:?}");
        }
    }

    #[test]
    fn count_matches_encode_length_for_edge_cases() {
        let encoding = tiktoken::get_encoding(O200K_BASE).unwrap();
        let cases = [
            "<|endoftext|>",
            "\u{200b}\u{200d}",
            "مرحبا بالعالم",
            "{\"number\":1234567890}",
            "fn main() { println!(\"hello\"); }",
        ];

        for text in cases {
            assert_eq!(count_o200k(text), encoding.encode(text).len());
        }
    }

    #[test]
    fn byte_ranges_balance_skewed_rows_without_reordering() {
        let values = [
            Some("a".to_owned()),
            None,
            Some("b".repeat(100)),
            Some("cc".to_owned()),
            Some("ddd".to_owned()),
            Some("eeee".to_owned()),
        ];
        let strings =
            StringChunked::from_iter_options("text".into(), values.iter().map(Option::as_deref));
        let ranges = byte_balanced_ranges(&strings, strings.get_values_size(), 3);

        assert_eq!(ranges.first().map(|range| range.0), Some(0));
        assert_eq!(
            ranges.iter().map(|range| range.1).sum::<usize>(),
            strings.len()
        );
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].0 + pair[0].1, pair[1].0);
        }
    }

    #[test]
    fn parallel_count_matches_sequential_with_nulls() {
        let values = (0..10_000).map(|index| match index % 5 {
            0 => None,
            1 => Some("short".to_owned()),
            2 => Some("你好，世界".repeat(20)),
            3 => Some("x".repeat(4_096)),
            _ => Some(format!("row {index}: 👋🏽")),
        });
        let strings = StringChunked::from_iter_options(
            "text".into(),
            values.collect::<Vec<_>>().iter().map(Option::as_deref),
        );

        let sequential = count_chunk(&strings).unwrap();
        let parallel = count_parallel(&strings, strings.get_values_size()).unwrap();
        assert!(sequential.into_iter().eq(&parallel));
    }
}
