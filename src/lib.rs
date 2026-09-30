//! Native count-only token expressions for Polars.

// The Gemma 3 feasibility kernel is intentionally not wired into the public
// Polars expression until full reference parity and artifact licensing are
// established.
#[cfg(feature = "gemma3-prototype")]
#[doc(hidden)]
pub mod sentencepiece_bpe;

use std::collections::{HashMap, VecDeque};

use polars::polars_utils::aliases::PlHashMap;
use polars::prelude::*;
use pyo3_polars::PolarsAllocator;
use pyo3_polars::derive::{CallerContext, polars_expr};
use pyo3_polars::export::polars_arrow::array::{Array, BinaryViewArray, PrimitiveArray, ValueSize};
use pyo3_polars::export::polars_core::{POOL, THREAD_POOL};
use rayon::prelude::*;
use serde::Deserialize;
use tiktoken::CoreBpe;

#[global_allocator]
static ALLOCATOR: PolarsAllocator = PolarsAllocator::new();

const O200K_BASE: &str = "o200k_base";
const CL100K_BASE: &str = "cl100k_base";
const P50K_BASE: &str = "p50k_base";
const R50K_BASE: &str = "r50k_base";
const SUPPORTED_TOKENIZERS: &str = "cl100k_base, o200k_base, p50k_base, r50k_base";
const PARALLEL_MIN_BYTES: usize = 512 * 1024;
const TASKS_PER_THREAD: usize = 4;
const CATEGORICAL_PARALLEL_MIN_CATEGORIES: usize = 4_096;
const CATEGORICAL_DENSE_RATIO: usize = 4;
const MAX_CACHE_CAPACITY: usize = 65_536;

#[derive(Deserialize)]
struct CountKwargs {
    tokenizer: String,
    #[serde(default)]
    cache_capacity: Option<usize>,
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

/// Count `p50k_base` tokens without materializing token IDs.
///
/// Special-token-looking byte sequences are treated as ordinary text.
///
/// # Panics
///
/// Panics only when the binary was built without the required
/// `vocab-p50k_base` Cargo feature, which is an internal build invariant.
#[inline]
#[must_use]
pub fn count_p50k(text: &str) -> usize {
    tiktoken::get_encoding(P50K_BASE)
        .expect("p50k_base vocabulary must be compiled in")
        .count(text)
}

/// Count `r50k_base` tokens without materializing token IDs.
///
/// Special-token-looking byte sequences are treated as ordinary text.
///
/// # Panics
///
/// Panics only when the binary was built without the required
/// `vocab-r50k_base` Cargo feature, which is an internal build invariant.
#[inline]
#[must_use]
pub fn count_r50k(text: &str) -> usize {
    tiktoken::get_encoding(R50K_BASE)
        .expect("r50k_base vocabulary must be compiled in")
        .count(text)
}

fn validate_tokenizer(tokenizer: &str) -> PolarsResult<()> {
    match tokenizer {
        CL100K_BASE | O200K_BASE | P50K_BASE | R50K_BASE => Ok(()),
        _ => Err(PolarsError::ComputeError(
            format!(
                "unsupported tokenizer {tokenizer:?}; supported tokenizers: {SUPPORTED_TOKENIZERS}"
            )
            .into(),
        )),
    }
}

fn resolve_encoding(tokenizer: &str) -> PolarsResult<&'static CoreBpe> {
    tiktoken::get_encoding(tokenizer).ok_or_else(|| {
        PolarsError::ComputeError(
            format!("tokenizer {tokenizer:?} was not compiled into this build").into(),
        )
    })
}

fn checked_count(text: &str, encoding: &CoreBpe) -> PolarsResult<u32> {
    if text.is_empty() {
        return Ok(0);
    }
    u32::try_from(encoding.count(text)).map_err(|_| {
        PolarsError::ComputeError("token count exceeds the UInt32 output range".into())
    })
}

fn count_chunk(strings: &StringChunked, encoding: &CoreBpe) -> PolarsResult<UInt32Chunked> {
    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(strings.name().clone(), strings.len());

    // Most string columns have no nulls. This iterator borrows StringView
    // values without boxing or producing an Option for every row.
    if strings.null_count() == 0 {
        for text in strings.into_no_null_iter() {
            output.append_value(checked_count(text, encoding)?);
        }
        return Ok(output.finish());
    }

    // Nullable iteration also borrows from StringView buffers; only the
    // output values and validity are allocated for this chunk.
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

#[inline]
fn cached_count<'a>(
    text: &'a str,
    encoding: &CoreBpe,
    capacity: usize,
    counts: &mut PlHashMap<&'a str, u32>,
    order: &mut VecDeque<&'a str>,
) -> PolarsResult<u32> {
    if text.is_empty() {
        return Ok(0);
    }
    if let Some(&count) = counts.get(text) {
        return Ok(count);
    }

    let count = checked_count(text, encoding)?;
    if counts.len() == capacity {
        if let Some(oldest) = order.pop_front() {
            counts.remove(oldest);
        }
    }
    counts.insert(text, count);
    order.push_back(text);
    Ok(count)
}

fn count_chunk_cached(
    strings: &StringChunked,
    encoding: &CoreBpe,
    capacity: usize,
) -> PolarsResult<UInt32Chunked> {
    debug_assert!(capacity > 0);
    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(strings.name().clone(), strings.len());
    // Borrow keys from the input buffers; the cache never copies string data.
    // FIFO eviction bounds both tables to `capacity` entries per task.
    let mut counts: PlHashMap<&str, u32> = PlHashMap::default();
    let mut order: VecDeque<&str> = VecDeque::new();

    if strings.null_count() == 0 {
        for text in strings.into_no_null_iter() {
            output.append_value(cached_count(
                text,
                encoding,
                capacity,
                &mut counts,
                &mut order,
            )?);
        }
        return Ok(output.finish());
    }

    for value in strings {
        match value {
            None => output.append_null(),
            Some(text) => output.append_value(cached_count(
                text,
                encoding,
                capacity,
                &mut counts,
                &mut order,
            )?),
        }
    }

    Ok(output.finish())
}

fn count_chunk_with_cache(
    strings: &StringChunked,
    encoding: &CoreBpe,
    cache_capacity: Option<usize>,
) -> PolarsResult<UInt32Chunked> {
    match cache_capacity {
        Some(capacity) => count_chunk_cached(strings, encoding, capacity),
        None => count_chunk(strings, encoding),
    }
}

fn binary_text(bytes: &[u8]) -> PolarsResult<&str> {
    std::str::from_utf8(bytes)
        .map_err(|_| PolarsError::ComputeError("Binary input contains invalid UTF-8".into()))
}

#[inline]
fn cached_binary_count<'a>(
    bytes: &'a [u8],
    encoding: &CoreBpe,
    capacity: usize,
    counts: &mut PlHashMap<&'a [u8], u32>,
    order: &mut VecDeque<&'a [u8]>,
) -> PolarsResult<u32> {
    if bytes.is_empty() {
        return Ok(0);
    }
    // A hit has already passed strict UTF-8 validation. Invalid bytes are
    // never inserted, so only misses need to be decoded.
    if let Some(&count) = counts.get(bytes) {
        return Ok(count);
    }
    let count = checked_count(binary_text(bytes)?, encoding)?;
    if counts.len() == capacity {
        if let Some(oldest) = order.pop_front() {
            counts.remove(oldest);
        }
    }
    counts.insert(bytes, count);
    order.push_back(bytes);
    Ok(count)
}

fn count_binary_chunk_with<'a>(
    binary: &'a BinaryChunked,
    mut count_bytes: impl FnMut(&'a [u8]) -> PolarsResult<u32>,
) -> PolarsResult<UInt32Chunked> {
    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(binary.name().clone(), binary.len());

    if binary.null_count() == 0 {
        for bytes in binary.into_no_null_iter() {
            output.append_value(count_bytes(bytes)?);
        }
    } else {
        for value in binary {
            match value {
                None => output.append_null(),
                Some(bytes) => output.append_value(count_bytes(bytes)?),
            }
        }
    }
    Ok(output.finish())
}

fn count_binary_chunk(
    binary: &BinaryChunked,
    encoding: &CoreBpe,
    cache_capacity: Option<usize>,
) -> PolarsResult<UInt32Chunked> {
    match cache_capacity {
        Some(capacity) => {
            let mut counts: PlHashMap<&[u8], u32> = PlHashMap::default();
            let mut order: VecDeque<&[u8]> = VecDeque::new();
            count_binary_chunk_with(binary, |bytes| {
                cached_binary_count(bytes, encoding, capacity, &mut counts, &mut order)
            })
        }
        None => {
            count_binary_chunk_with(binary, |bytes| checked_count(binary_text(bytes)?, encoding))
        }
    }
}

fn empty_binary_with_nulls(binary: &BinaryChunked) -> UInt32Chunked {
    let arrays = binary.downcast_iter().map(|chunk| {
        PrimitiveArray::<u32>::from_vec(vec![0; chunk.len()])
            .with_validity(chunk.validity().cloned())
    });
    UInt32Chunked::from_chunk_iter(binary.name().clone(), arrays)
}

fn constant_strings_with_nulls(strings: &StringChunked, count: u32) -> UInt32Chunked {
    // Reuse each input chunk's null bitmap rather than rebuilding validity.
    let arrays = strings.downcast_iter().map(|chunk| {
        PrimitiveArray::<u32>::from_vec(vec![count; chunk.len()])
            .with_validity(chunk.validity().cloned())
    });
    UInt32Chunked::from_chunk_iter(strings.name().clone(), arrays)
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
    let mut count_category = |category: T::Native| -> PolarsResult<u32> {
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
        Ok(counts[index])
    };

    if physical.null_count() == 0 {
        let mut values = Vec::with_capacity(physical.len());
        for category in physical.into_no_null_iter() {
            values.push(count_category(category)?);
        }
        return Ok(UInt32Chunked::from_vec(physical.name().clone(), values));
    }

    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(physical.name().clone(), physical.len());
    for value in physical {
        match value {
            None => output.append_null(),
            Some(category) => output.append_value(count_category(category)?),
        }
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
fn byte_balanced_ranges_from_lengths(
    len: usize,
    total_bytes: usize,
    max_parts: usize,
    lengths: impl Iterator<Item = usize>,
) -> (Vec<(usize, usize)>, usize) {
    debug_assert!(max_parts > 0);
    if len == 0 {
        return (Vec::new(), 0);
    }

    let mut ranges = Vec::with_capacity(max_parts.min(len));
    let mut start = 0;
    let mut range_bytes = 0;
    let mut remaining_bytes = total_bytes;
    let mut remaining_parts = max_parts.min(len);
    let mut largest_row_bytes = 0;

    let mut index = 0;
    let mut add_row = |row_bytes: usize| {
        range_bytes += row_bytes;
        largest_row_bytes = largest_row_bytes.max(row_bytes);
        let target = remaining_bytes.div_ceil(remaining_parts);
        let rows_remaining = len - index - 1;

        if remaining_parts > 1 && range_bytes >= target && rows_remaining > 0 {
            ranges.push((start, index + 1 - start));
            start = index + 1;
            remaining_bytes = remaining_bytes.saturating_sub(range_bytes);
            remaining_parts -= 1;
            range_bytes = 0;
        }
        index += 1;
    };

    for row_bytes in lengths {
        add_row(row_bytes);
    }
    debug_assert_eq!(index, len);

    if start < len {
        ranges.push((start, len - start));
    }
    (ranges, largest_row_bytes)
}

fn byte_balanced_ranges(
    strings: &StringChunked,
    total_bytes: usize,
    max_parts: usize,
) -> (Vec<(usize, usize)>, usize) {
    // Length is stored directly in each view. Reading it avoids touching
    // string payload buffers merely to schedule tokenization work.
    let lengths = strings.downcast_iter().flat_map(|chunk| {
        let validity = chunk.validity();
        chunk.views().iter().enumerate().map(move |(offset, view)| {
            if validity.is_none_or(|bitmap| bitmap.get_bit(offset)) {
                view.length as usize
            } else {
                0
            }
        })
    });
    byte_balanced_ranges_from_lengths(strings.len(), total_bytes, max_parts, lengths)
}

fn binary_byte_balanced_ranges(
    binary: &BinaryChunked,
    total_bytes: usize,
    max_parts: usize,
) -> (Vec<(usize, usize)>, usize) {
    let lengths = binary.downcast_iter().flat_map(|chunk| {
        let validity = chunk.validity();
        chunk.views().iter().enumerate().map(move |(offset, view)| {
            if validity.is_none_or(|bitmap| bitmap.get_bit(offset)) {
                view.length as usize
            } else {
                0
            }
        })
    });
    byte_balanced_ranges_from_lengths(binary.len(), total_bytes, max_parts, lengths)
}

fn count_parallel(
    strings: &StringChunked,
    total_bytes: usize,
    encoding: &CoreBpe,
    cache_capacity: Option<usize>,
) -> PolarsResult<UInt32Chunked> {
    let n_threads = THREAD_POOL.current_num_threads();
    // More partitions help balance uncached rows with uneven tokenization
    // costs. Cached partitions each need to warm their own lookup, so keep
    // one partition per worker when caching is enabled.
    let max_parts = if cache_capacity.is_some() {
        n_threads.max(1)
    } else {
        n_threads.saturating_mul(TASKS_PER_THREAD).max(1)
    };
    let (ranges, largest_row_bytes) = byte_balanced_ranges(strings, total_bytes, max_parts);
    // A dominant row cannot be split. If the other rows are too small to
    // amortize Rayon dispatch and output concatenation, stay sequential.
    if total_bytes.saturating_sub(largest_row_bytes) < PARALLEL_MIN_BYTES {
        return count_chunk_with_cache(strings, encoding, cache_capacity);
    }

    let chunks = POOL.install(|| {
        ranges
            .into_par_iter()
            .map(|(offset, len)| {
                let offset = i64::try_from(offset).map_err(|_| {
                    PolarsError::ComputeError("row offset exceeds the Int64 range".into())
                })?;
                count_chunk_with_cache(&strings.slice(offset, len), encoding, cache_capacity)
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

fn count_binary_parallel(
    binary: &BinaryChunked,
    total_bytes: usize,
    encoding: &CoreBpe,
    cache_capacity: Option<usize>,
) -> PolarsResult<UInt32Chunked> {
    let n_threads = THREAD_POOL.current_num_threads();
    let max_parts = if cache_capacity.is_some() {
        n_threads.max(1)
    } else {
        n_threads.saturating_mul(TASKS_PER_THREAD).max(1)
    };
    let (ranges, largest_row_bytes) = binary_byte_balanced_ranges(binary, total_bytes, max_parts);
    if total_bytes.saturating_sub(largest_row_bytes) < PARALLEL_MIN_BYTES {
        return count_binary_chunk(binary, encoding, cache_capacity);
    }

    let chunks = POOL.install(|| {
        ranges
            .into_par_iter()
            .map(|(offset, len)| {
                let offset = i64::try_from(offset).map_err(|_| {
                    PolarsError::ComputeError("row offset exceeds the Int64 range".into())
                })?;
                count_binary_chunk(&binary.slice(offset, len), encoding, cache_capacity)
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;
    let arrays = chunks
        .into_iter()
        .flat_map(|chunk| chunk.downcast_iter().cloned().collect::<Vec<_>>());
    Ok(UInt32Chunked::from_chunk_iter(
        binary.name().clone(),
        arrays,
    ))
}

fn count_string_input(
    strings: &StringChunked,
    tokenizer: &str,
    cache_capacity: Option<usize>,
    allow_parallel: bool,
) -> PolarsResult<UInt32Chunked> {
    let null_count = strings.null_count();
    if null_count == strings.len() {
        // A null StringView may still reference non-empty bytes from an
        // earlier value. Check validity before physical byte size.
        return Ok(UInt32Chunked::full_null(
            strings.name().clone(),
            strings.len(),
        ));
    }

    let physical_bytes = strings.get_values_size();
    // Null StringViews may retain earlier byte lengths. When nearly every
    // row is null, sum only valid view lengths so those dead bytes do not
    // trigger or skew parallel work.
    let total_bytes = if physical_bytes >= PARALLEL_MIN_BYTES
        && null_count >= strings.len() - strings.len() / 8
    {
        strings
            .downcast_iter()
            .flat_map(|chunk| chunk.non_null_views_iter())
            .map(|view| view.length as usize)
            .sum()
    } else {
        physical_bytes
    };
    if total_bytes == 0 {
        // Every non-null value is empty. Mixed empty/null input still needs
        // its validity copied to the output.
        return Ok(if null_count == 0 {
            UInt32Chunked::full(strings.name().clone(), 0, strings.len())
        } else {
            constant_strings_with_nulls(strings, 0)
        });
    }

    if strings.len() >= 1_024 && null_count <= strings.len() / 8 {
        if null_count == 0 {
            let mut values = strings.into_no_null_iter();
            if let Some(first) = values.next()
                && values.next().is_some_and(|second| second == first)
                && strings.get(strings.len() / 2) == Some(first)
                && strings.get(strings.len() - 1) == Some(first)
                && values.all(|text| text == first)
            {
                let count = checked_count(first, resolve_encoding(tokenizer)?)?;
                return Ok(UInt32Chunked::full(
                    strings.name().clone(),
                    count,
                    strings.len(),
                ));
            }
        } else {
            let mut values = strings.into_iter().flatten();
            if let Some(first) = values.next()
                && values.next().is_some_and(|second| second == first)
                && strings
                    .get(strings.len() / 2)
                    .is_none_or(|text| text == first)
                && strings
                    .get(strings.len() - 1)
                    .is_none_or(|text| text == first)
                && values.all(|text| text == first)
            {
                let count = checked_count(first, resolve_encoding(tokenizer)?)?;
                return Ok(constant_strings_with_nulls(strings, count));
            }
        }
    }

    let encoding = resolve_encoding(tokenizer)?;
    if allow_parallel && total_bytes >= PARALLEL_MIN_BYTES {
        count_parallel(strings, total_bytes, encoding, cache_capacity)
    } else {
        count_chunk_with_cache(strings, encoding, cache_capacity)
    }
}

fn count_binary_input(
    binary: &BinaryChunked,
    tokenizer: &str,
    cache_capacity: Option<usize>,
    allow_parallel: bool,
) -> PolarsResult<UInt32Chunked> {
    let null_count = binary.null_count();
    if null_count == binary.len() {
        return Ok(UInt32Chunked::full_null(
            binary.name().clone(),
            binary.len(),
        ));
    }

    let physical_bytes: usize = binary
        .downcast_iter()
        .map(BinaryViewArray::total_bytes_len)
        .sum();
    let total_bytes =
        if physical_bytes >= PARALLEL_MIN_BYTES && null_count >= binary.len() - binary.len() / 8 {
            binary
                .downcast_iter()
                .flat_map(|chunk| chunk.non_null_views_iter())
                .map(|view| view.length as usize)
                .sum()
        } else {
            physical_bytes
        };
    if total_bytes == 0 {
        return Ok(if null_count == 0 {
            UInt32Chunked::full(binary.name().clone(), 0, binary.len())
        } else {
            empty_binary_with_nulls(binary)
        });
    }

    let encoding = resolve_encoding(tokenizer)?;
    if allow_parallel && total_bytes >= PARALLEL_MIN_BYTES {
        count_binary_parallel(binary, total_bytes, encoding, cache_capacity)
    } else {
        count_binary_chunk(binary, encoding, cache_capacity)
    }
}

#[polars_expr(output_type=UInt32)]
#[allow(clippy::needless_pass_by_value)] // Required by the plugin macro ABI.
fn token_count(
    inputs: &[Series],
    context: CallerContext,
    kwargs: CountKwargs,
) -> PolarsResult<Series> {
    // Keep invalid tokenizer errors ahead of constant-input fast paths while
    // deferring vocabulary initialization until there is text to tokenize.
    validate_tokenizer(&kwargs.tokenizer)?;
    if let Some(capacity) = kwargs.cache_capacity {
        if !(1..=MAX_CACHE_CAPACITY).contains(&capacity) {
            return Err(PolarsError::ComputeError(
                format!("cache_capacity must be between 1 and {MAX_CACHE_CAPACITY}").into(),
            ));
        }
    }

    let input = &inputs[0];
    let allow_parallel = !context.parallel() && THREAD_POOL.current_num_threads() > 1;
    let output = match input.dtype() {
        DataType::String => count_string_input(
            input.str()?,
            &kwargs.tokenizer,
            kwargs.cache_capacity,
            allow_parallel,
        )?,
        DataType::Binary => count_binary_input(
            input.binary()?,
            &kwargs.tokenizer,
            kwargs.cache_capacity,
            allow_parallel,
        )?,
        DataType::Categorical(_, _) | DataType::Enum(_, _) => {
            if input.null_count() == input.len() {
                // No category IDs are used, so neither the mapping nor a
                // count lookup needs to be allocated or traversed.
                UInt32Chunked::full_null(input.name().clone(), input.len())
            } else {
                let encoding = resolve_encoding(&kwargs.tokenizer)?;
                with_match_categorical_physical_type!(input.dtype().cat_physical()?, |$C| {
                    count_categorical(input.cat::<$C>()?, encoding, allow_parallel)
                })?
            }
        }
        dtype => {
            return Err(PolarsError::ComputeError(
                format!("expected `String`, `Binary`, `Categorical`, or `Enum`, got {dtype}")
                    .into(),
            ));
        }
    };

    Ok(output.into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_validation_does_not_resolve_vocabulary() {
        for tokenizer in [CL100K_BASE, O200K_BASE, P50K_BASE, R50K_BASE] {
            assert!(validate_tokenizer(tokenizer).is_ok());
        }
        assert!(validate_tokenizer("unsupported").is_err());
    }

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
    fn legacy_encodings_match_encode_length() {
        let cases = ["", "hello world", "    code", "你好，世界", "<|endoftext|>"];
        for name in [P50K_BASE, R50K_BASE] {
            let encoding = resolve_encoding(name).unwrap();
            for text in cases {
                assert_eq!(
                    encoding.count(text),
                    encoding.encode(text).len(),
                    "{name}: {text:?}"
                );
                let public_count = match name {
                    P50K_BASE => count_p50k(text),
                    R50K_BASE => count_r50k(text),
                    _ => unreachable!(),
                };
                assert_eq!(public_count, encoding.count(text), "{name}: {text:?}");
            }
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
        let (ranges, largest_row_bytes) =
            byte_balanced_ranges(&strings, strings.get_values_size(), 3);

        assert_eq!(largest_row_bytes, 100);
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
    fn dominant_row_uses_sequential_fallback_with_exact_counts() {
        let values = [
            Some("你好👋".repeat(100_000)),
            None,
            Some("short".to_owned()),
        ];
        let strings =
            StringChunked::from_iter_options("text".into(), values.iter().map(Option::as_deref));
        let total_bytes = strings.get_values_size();
        let (_, largest_row_bytes) = byte_balanced_ranges(&strings, total_bytes, 4);
        assert!(total_bytes - largest_row_bytes < PARALLEL_MIN_BYTES);

        let encoding = resolve_encoding(O200K_BASE).unwrap();
        let sequential = count_chunk(&strings, encoding).unwrap();
        let parallel = count_parallel(&strings, total_bytes, encoding, None).unwrap();
        assert!(sequential.into_iter().eq(&parallel));
        let cached = count_parallel(&strings, total_bytes, encoding, Some(2)).unwrap();
        assert!(parallel.into_iter().eq(&cached));
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
        let parallel = count_parallel(&strings, strings.get_values_size(), encoding, None).unwrap();
        assert!(sequential.into_iter().eq(&parallel));
        let cached =
            count_parallel(&strings, strings.get_values_size(), encoding, Some(256)).unwrap();
        assert!(parallel.into_iter().eq(&cached));
    }
}
