//! Isolated Polars string/output benchmark for an experimental whole-value cache.

use std::collections::{HashMap, VecDeque};
use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use polars::prelude::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use tiktoken::CoreBpe;

const ROWS: usize = 100_000;
const TARGET_BYTES: usize = 128;
const SAMPLE: &str = "The quick brown fox jumps over 13 lazy dogs. 你好👋 ";
const MODE_ENV: &str = "POLARS_TOKENIZER_CACHE_POLARS_MODE";
const TOKENIZER_ENV: &str = "POLARS_TOKENIZER_CACHE_POLARS_TOKENIZER";
const UNIQUE_ENV: &str = "POLARS_TOKENIZER_CACHE_POLARS_UNIQUE";
const NULL_ROWS_ENV: &str = "POLARS_TOKENIZER_CACHE_POLARS_NULL_ROWS";
const REPEATS_ENV: &str = "POLARS_TOKENIZER_CACHE_POLARS_REPEATS";

fn value(index: usize) -> String {
    let suffix = format!(" [{index}]");
    let mut text = String::with_capacity(TARGET_BYTES);
    while text.len() + SAMPLE.len() + suffix.len() <= TARGET_BYTES {
        text.push_str(SAMPLE);
    }
    while text.len() + suffix.len() < TARGET_BYTES {
        text.push('x');
    }
    text.push_str(&suffix);
    text
}

fn make_input(unique: usize, null_rows: usize) -> StringChunked {
    assert!(null_rows < ROWS);
    let non_null_rows = ROWS - null_rows;
    assert!((1..=non_null_rows).contains(&unique));
    let dictionary: Vec<String> = (0..unique).map(value).collect();
    assert!(dictionary.iter().all(|text| text.len() == TARGET_BYTES));
    let mut indices: Vec<Option<usize>> = (0..non_null_rows)
        .map(|index| Some(index % unique))
        .collect();
    indices.extend((0..null_rows).map(|_| None));

    let mut state = 42_usize;
    for index in (1..indices.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        indices.swap(index, state % (index + 1));
    }

    StringChunked::from_iter_options(
        "text".into(),
        indices
            .iter()
            .map(|index| index.map(|index| dictionary[index].as_str())),
    )
}

fn count_uncached(strings: &StringChunked, encoding: &CoreBpe) -> UInt32Chunked {
    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(strings.name().clone(), strings.len());
    for value in strings {
        match value {
            None => output.append_null(),
            Some(text) => output.append_value(
                u32::try_from(encoding.count(text)).expect("count fits UInt32 in this corpus"),
            ),
        }
    }
    output.finish()
}

fn count_cached(strings: &StringChunked, encoding: &CoreBpe, capacity: usize) -> UInt32Chunked {
    let mut cache: HashMap<&str, u32> = HashMap::new();
    let mut insertion_order = VecDeque::new();
    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(strings.name().clone(), strings.len());
    for value in strings {
        let Some(text) = value else {
            output.append_null();
            continue;
        };
        if let Some(&count) = cache.get(text) {
            output.append_value(count);
            continue;
        }
        let count = u32::try_from(encoding.count(text)).expect("count fits UInt32 in this corpus");
        if cache.len() == capacity {
            let oldest = insertion_order
                .pop_front()
                .expect("full cache has an entry");
            cache.remove(oldest);
        }
        cache.insert(text, count);
        insertion_order.push_back(text);
        output.append_value(count);
    }
    output.finish()
}

fn digest_strings(strings: &StringChunked) -> String {
    let mut digest = Sha256::new();
    for value in strings {
        match value {
            None => digest.update(b"N"),
            Some(text) => {
                digest.update(b"S");
                digest.update(
                    u64::try_from(text.len())
                        .expect("value length fits u64")
                        .to_le_bytes(),
                );
                digest.update(text.as_bytes());
            }
        }
    }
    format!("{:x}", digest.finalize())
}

fn digest_counts(counts: &UInt32Chunked) -> (String, u64) {
    let mut digest = Sha256::new();
    let mut total = 0_u64;
    for value in counts {
        match value {
            None => digest.update(b"N"),
            Some(count) => {
                digest.update(b"C");
                digest.update(count.to_le_bytes());
                total += u64::from(count);
            }
        }
    }
    (format!("{:x}", digest.finalize()), total)
}

fn linux_rss_bytes(field: &str) -> Option<usize> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with(field))?;
    let kib = line.split_whitespace().nth(1)?.parse::<usize>().ok()?;
    kib.checked_mul(1_024)
}

fn run() {
    let mode = env::var(MODE_ENV).expect("cache mode must be selected by the driver");
    let tokenizer = env::var(TOKENIZER_ENV).expect("tokenizer must be selected by the driver");
    let unique = env::var(UNIQUE_ENV)
        .expect("unique count must be selected by the driver")
        .parse::<usize>()
        .expect("unique count must be an integer");
    let null_rows = env::var(NULL_ROWS_ENV)
        .unwrap_or_else(|_| "0".to_owned())
        .parse::<usize>()
        .expect("null rows must be an integer");
    let repeats = env::var(REPEATS_ENV)
        .unwrap_or_else(|_| "5".to_owned())
        .parse::<usize>()
        .expect("repeats must be an integer");
    assert!(repeats > 0);
    assert!(matches!(tokenizer.as_str(), "o200k_base" | "cl100k_base"));
    assert!(matches!(
        mode.as_str(),
        "uncached" | "cache_256" | "cache_4096"
    ));
    let encoding = tiktoken::get_encoding(&tokenizer).expect("vocabulary must be compiled in");
    let strings = make_input(unique, null_rows);
    let bytes: usize = strings.into_iter().flatten().map(str::len).sum();
    let dataset_sha256 = digest_strings(&strings);
    let rss_before_bytes = linux_rss_bytes("VmRSS:");

    let operation = || match mode.as_str() {
        "uncached" => count_uncached(&strings, encoding),
        "cache_256" => count_cached(&strings, encoding, 256),
        "cache_4096" => count_cached(&strings, encoding, 4_096),
        _ => unreachable!("validated mode"),
    };
    black_box(operation());
    let mut samples = Vec::with_capacity(repeats);
    let mut result = None;
    for _ in 0..repeats {
        let start = Instant::now();
        let output = black_box(operation());
        samples.push(start.elapsed().as_secs_f64());
        result = Some(output);
    }
    let rss_after_bytes = linux_rss_bytes("VmRSS:");
    let peak_rss_bytes = linux_rss_bytes("VmHWM:");
    let (counts_sha256, total_tokens) = digest_counts(&result.expect("at least one repeat"));
    let mut ordered_samples = samples.clone();
    ordered_samples.sort_by(f64::total_cmp);
    let middle = ordered_samples.len() / 2;
    let median_seconds = if ordered_samples.len().is_multiple_of(2) {
        f64::midpoint(ordered_samples[middle - 1], ordered_samples[middle])
    } else {
        ordered_samples[middle]
    };
    let report = json!({
        "schema_version": 1,
        "mode": mode,
        "tokenizer": tokenizer,
        "rows": ROWS,
        "unique_values": unique,
        "null_rows": null_rows,
        "bytes": bytes,
        "dataset_sha256": dataset_sha256,
        "counts_sha256": counts_sha256,
        "total_tokens": total_tokens,
        "samples_seconds": samples,
        "median_seconds": median_seconds,
        "mib_per_second": f64::from(u32::try_from(bytes).expect("corpus bytes fit u32"))
            / median_seconds / (1024.0 * 1024.0),
        "rss_before_bytes": rss_before_bytes,
        "rss_after_bytes": rss_after_bytes,
        "peak_rss_bytes": peak_rss_bytes,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize benchmark report")
    );
}

fn main() {
    if env::var_os(MODE_ENV).is_none() {
        eprintln!("skipping cache_polars benchmark: use the Python matrix driver");
        return;
    }
    run();
}
