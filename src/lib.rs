//! Native count-only token expressions for Polars.

// The Gemma 3 feasibility kernel is intentionally not wired into the public
// Polars expression until full reference parity and artifact licensing are
// established.
#[cfg(feature = "gemma3-prototype")]
#[doc(hidden)]
pub mod sentencepiece_bpe;

use std::collections::HashMap;

use polars::prelude::*;
use pyo3_polars::PolarsAllocator;
use pyo3_polars::derive::{CallerContext, polars_expr};
use pyo3_polars::export::polars_arrow::array::ValueSize;
use pyo3_polars::export::polars_core::{POOL, THREAD_POOL};
use rayon::prelude::*;
use serde::Deserialize;
use tiktoken::CoreBpe;

#[global_allocator]
static ALLOCATOR: PolarsAllocator = PolarsAllocator::new();

const O200K_BASE: &str = "o200k_base";
const CL100K_BASE: &str = "cl100k_base";
const SUPPORTED_TOKENIZERS: &str = "cl100k_base, o200k_base";
const PARALLEL_MIN_BYTES: usize = 512 * 1024;
const TASKS_PER_THREAD: usize = 4;
const CATEGORICAL_PARALLEL_MIN_CATEGORIES: usize = 4_096;
const CATEGORICAL_DENSE_RATIO: usize = 4;

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

/// Count `cl100k_base` tokens without materializing token IDs.
///
/// Special-token-looking byte sequences are treated as ordinary text. The
/// vocabulary is initialized once by the `tiktoken` crate and then shared.
///
/// # Panics
///
/// Panics only when the binary was built without the required
/// `vocab-cl100k_base` Cargo feature, which is an internal build invariant.
#[inline]
#[must_use]
pub fn count_cl100k(text: &str) -> usize {
    tiktoken::get_encoding(CL100K_BASE)
        .expect("cl100k_base vocabulary must be compiled in")
        .count(text)
}

fn resolve_encoding(tokenizer: &str) -> PolarsResult<&'static CoreBpe> {
    match tokenizer {
        CL100K_BASE | O200K_BASE => tiktoken::get_encoding(tokenizer).ok_or_else(|| {
            PolarsError::ComputeError(
                format!("tokenizer {tokenizer:?} was not compiled into this build").into(),
            )
        }),
        _ => Err(PolarsError::ComputeError(
            format!(
                "unsupported tokenizer {tokenizer:?}; supported tokenizers: {SUPPORTED_TOKENIZERS}"
            )
            .into(),
        )),
    }
}

fn checked_count(text: &str, encoding: &CoreBpe) -> PolarsResult<u32> {
    u32::try_from(encoding.count(text)).map_err(|_| {
        PolarsError::ComputeError("token count exceeds the UInt32 output range".into())
    })
}

fn count_chunk(strings: &StringChunked, encoding: &CoreBpe) -> PolarsResult<UInt32Chunked> {
    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(strings.name().clone(), strings.len());

    // Iteration borrows values from Polars' StringView buffers. The only
    // per-chunk allocation here is the final values/validity output.
    for value in strings {
        match value {
            None => output.append_null(),
            Some(text) => {
                output.append_value(checked_count(text, encoding)?);
            }
        }
    }

    Ok(output.finish())
}

fn categorical_output<T: PolarsCategoricalType>(
    categorical: &CategoricalChunked<T>,
    counts: &[u32],
) -> PolarsResult<UInt32Chunked> {
    let physical = categorical.physical();
    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(physical.name().clone(), physical.len());

    for value in physical {
        let Some(category) = value else {
            output.append_null();
            continue;
        };
        let category = category.as_cat();
        let index = category as usize;
        let count = counts.get(index).ok_or_else(|| {
            PolarsError::ComputeError("categorical index exceeds its count lookup".into())
        })?;
        output.append_value(*count);
    }

    Ok(output.finish())
}

fn count_categorical_dense<T: PolarsCategoricalType>(
    categorical: &CategoricalChunked<T>,
    encoding: &CoreBpe,
) -> PolarsResult<UInt32Chunked> {
    let physical = categorical.physical();
    let mapping = categorical.get_mapping();
    let mapping_len = mapping.num_cats_upper_bound();
    let mut counts = vec![0_u32; mapping_len];
    let mut seen = vec![false; mapping_len];
    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(physical.name().clone(), physical.len());

    for value in physical {
        let Some(category) = value else {
            output.append_null();
            continue;
        };
        let category = category.as_cat();
        let index = category as usize;
        if index >= mapping_len {
            return Err(PolarsError::ComputeError(
                "categorical index exceeds its string mapping".into(),
            ));
        }
        if !seen[index] {
            let text = mapping.cat_to_str(category).ok_or_else(|| {
                PolarsError::ComputeError("categorical value is missing from its mapping".into())
            })?;
            counts[index] = checked_count(text, encoding)?;
            seen[index] = true;
        }
        output.append_value(counts[index]);
    }

    Ok(output.finish())
}

fn count_categorical_sparse<T: PolarsCategoricalType>(
    categorical: &CategoricalChunked<T>,
    encoding: &CoreBpe,
) -> PolarsResult<UInt32Chunked> {
    let physical = categorical.physical();
    let mapping = categorical.get_mapping();
    let mut counts = HashMap::with_capacity(physical.len().min(4_096));
    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(physical.name().clone(), physical.len());

    for value in physical {
        let Some(category) = value else {
            output.append_null();
            continue;
        };
        let category = category.as_cat();
        let count = if let Some(count) = counts.get(&category) {
            *count
        } else {
            let text = mapping.cat_to_str(category).ok_or_else(|| {
                PolarsError::ComputeError("categorical value is missing from its mapping".into())
            })?;
            let count = checked_count(text, encoding)?;
            counts.insert(category, count);
            count
        };
        output.append_value(count);
    }

    Ok(output.finish())
}

fn count_categorical_parallel<T: PolarsCategoricalType>(
    categorical: &CategoricalChunked<T>,
    encoding: &CoreBpe,
) -> PolarsResult<UInt32Chunked> {
    let physical = categorical.physical();
    let mapping = categorical.get_mapping();
    let mapping_len = mapping.num_cats_upper_bound();
    let mut seen = vec![false; mapping_len];
    let mut used = Vec::with_capacity(mapping_len.min(physical.len()));
    for category in physical.into_iter().flatten() {
        let category = category.as_cat();
        let index = category as usize;
        if index >= mapping_len {
            return Err(PolarsError::ComputeError(
                "categorical index exceeds its string mapping".into(),
            ));
        }
        if !seen[index] {
            seen[index] = true;
            used.push(category);
        }
    }

    let total_bytes = used.iter().try_fold(0_usize, |total, category| {
        mapping
            .cat_to_str(*category)
            .map(|text| total.saturating_add(text.len()))
            .ok_or_else(|| {
                PolarsError::ComputeError("categorical value is missing from its mapping".into())
            })
    })?;
    if total_bytes < PARALLEL_MIN_BYTES {
        return count_categorical_dense(categorical, encoding);
    }

    let token_counts = POOL.install(|| {
        used.par_iter()
            .map(|category| {
                let text = mapping.cat_to_str(*category).ok_or_else(|| {
                    PolarsError::ComputeError(
                        "categorical value is missing from its mapping".into(),
                    )
                })?;
                checked_count(text, encoding)
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;
    let mut counts = vec![0_u32; mapping_len];
    for (category, count) in used.into_iter().zip(token_counts) {
        counts[category as usize] = count;
    }
    categorical_output(categorical, &counts)
}

fn count_categorical<T: PolarsCategoricalType>(
    categorical: &CategoricalChunked<T>,
    encoding: &CoreBpe,
    allow_parallel: bool,
) -> PolarsResult<UInt32Chunked> {
    let mapping_len = categorical.get_mapping().num_cats_upper_bound();
    let dense_limit = categorical
        .len()
        .saturating_mul(CATEGORICAL_DENSE_RATIO)
        .max(1_024);
    if mapping_len > dense_limit {
        count_categorical_sparse(categorical, encoding)
    } else if allow_parallel && mapping_len >= CATEGORICAL_PARALLEL_MIN_CATEGORIES {
        count_categorical_parallel(categorical, encoding)
    } else {
        count_categorical_dense(categorical, encoding)
    }
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

fn count_parallel(
    strings: &StringChunked,
    total_bytes: usize,
    encoding: &CoreBpe,
) -> PolarsResult<UInt32Chunked> {
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
                count_chunk(&strings.slice(offset, len), encoding)
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
    let encoding = resolve_encoding(&kwargs.tokenizer)?;

    let input = &inputs[0];
    let output = match input.dtype() {
        DataType::String => {
            let strings = input.str()?;
            let total_bytes = strings.get_values_size();
            let should_parallelize = !context.parallel()
                && THREAD_POOL.current_num_threads() > 1
                && total_bytes >= PARALLEL_MIN_BYTES;
            if should_parallelize {
                count_parallel(strings, total_bytes, encoding)?
            } else {
                count_chunk(strings, encoding)?
            }
        }
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            with_match_categorical_physical_type!(input.dtype().cat_physical()?, |$C| {
                count_categorical(
                    input.cat::<$C>()?,
                    encoding,
                    !context.parallel() && THREAD_POOL.current_num_threads() > 1,
                )
            })?
        }
        dtype => {
            return Err(PolarsError::ComputeError(
                format!("expected `String`, `Categorical`, or `Enum`, got {dtype}").into(),
            ));
        }
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
    fn known_cl100k_counts() {
        let cases = [
            ("", 0),
            ("hello world", 2),
            ("hello\0world", 3),
            ("你好，世界", 6),
            ("e\u{301}", 2),
        ];

        for (text, expected) in cases {
            assert_eq!(count_cl100k(text), expected, "input: {text:?}");
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

        let encoding = resolve_encoding(O200K_BASE).unwrap();
        let sequential = count_chunk(&strings, encoding).unwrap();
        let parallel = count_parallel(&strings, strings.get_values_size(), encoding).unwrap();
        assert!(sequential.into_iter().eq(&parallel));
    }
}
