//! Isolated comparisons of exact counting, Polars input iteration, and Arrow output.

use std::env;
use std::fs;
use std::hint::black_box;
use std::time::Instant;

use polars::prelude::*;
use pyo3_polars::export::polars_arrow::array::ValueSize;
use serde_json::json;
use sha2::{Digest, Sha256};
use tiktoken::CoreBpe;

const ROWS: usize = 100_000;
const TARGET_BYTES: usize = 128;
const SAMPLE: &str = "The quick brown fox jumps over 13 lazy dogs. 你好👋 ";
const MODE_ENV: &str = "POLARS_TOKENIZER_COMPONENT_MODE";
const TOKENIZER_ENV: &str = "POLARS_TOKENIZER_COMPONENT_TOKENIZER";
const REPEATS_ENV: &str = "POLARS_TOKENIZER_COMPONENT_REPEATS";

enum BenchmarkOutput {
    Vector(Vec<Option<u32>>),
    Arrow(UInt32Chunked),
}

fn make_value(index: usize) -> String {
    let suffix = format!(" [{index}]");
    let mut value = String::with_capacity(TARGET_BYTES);
    while value.len() + SAMPLE.len() + suffix.len() <= TARGET_BYTES {
        value.push_str(SAMPLE);
    }
    while value.len() + suffix.len() < TARGET_BYTES {
        value.push('x');
    }
    value.push_str(&suffix);
    assert_eq!(value.len(), TARGET_BYTES);
    value
}

fn make_input() -> Vec<Option<String>> {
    (0..ROWS)
        .map(|index| (index % 100 != 0).then(|| make_value(index)))
        .collect()
}

fn count_value(text: &str, encoding: &CoreBpe) -> u32 {
    u32::try_from(encoding.count(text)).expect("benchmark counts fit UInt32")
}

fn kernel_vec(values: &[Option<String>], encoding: &CoreBpe) -> Vec<Option<u32>> {
    values
        .iter()
        .map(|value| value.as_deref().map(|text| count_value(text, encoding)))
        .collect()
}

fn column_vec(strings: &StringChunked, encoding: &CoreBpe) -> Vec<Option<u32>> {
    strings
        .into_iter()
        .map(|value| value.map(|text| count_value(text, encoding)))
        .collect()
}

fn arrow_output(values: impl IntoIterator<Item = Option<u32>>) -> UInt32Chunked {
    let mut output = PrimitiveChunkedBuilder::<UInt32Type>::new("text".into(), ROWS);
    for value in values {
        match value {
            Some(count) => output.append_value(count),
            None => output.append_null(),
        }
    }
    output.finish()
}

fn column_arrow(strings: &StringChunked, encoding: &CoreBpe) -> UInt32Chunked {
    arrow_output(
        strings
            .into_iter()
            .map(|value| value.map(|text| count_value(text, encoding))),
    )
}

fn digest_input(values: &[Option<String>]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        match value {
            None => digest.update(b"N"),
            Some(text) => {
                digest.update(b"S");
                digest.update(
                    u64::try_from(text.len())
                        .expect("length fits u64")
                        .to_le_bytes(),
                );
                digest.update(text.as_bytes());
            }
        }
    }
    format!("{:x}", digest.finalize())
}

fn digest_counts(values: impl IntoIterator<Item = Option<u32>>) -> (String, u64) {
    let mut digest = Sha256::new();
    let mut total = 0_u64;
    for value in values {
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
    line.split_whitespace()
        .nth(1)?
        .parse::<usize>()
        .ok()?
        .checked_mul(1_024)
}

fn median(values: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    if ordered.len().is_multiple_of(2) {
        f64::midpoint(ordered[middle - 1], ordered[middle])
    } else {
        ordered[middle]
    }
}

fn run() {
    let mode = env::var(MODE_ENV).expect("mode must be selected by the driver");
    let tokenizer = env::var(TOKENIZER_ENV).expect("tokenizer must be selected by the driver");
    let repeats = env::var(REPEATS_ENV)
        .unwrap_or_else(|_| "5".to_owned())
        .parse::<usize>()
        .expect("repeats must be an integer");
    assert!(repeats > 0);
    assert!(matches!(tokenizer.as_str(), "o200k_base" | "cl100k_base"));
    assert!(matches!(
        mode.as_str(),
        "kernel_vec" | "column_vec" | "column_arrow" | "output_only"
    ));

    let encoding = tiktoken::get_encoding(&tokenizer).expect("vocabulary must be compiled in");
    let values = make_input();
    let strings = StringChunked::from_iter_options(
        "text".into(),
        values.iter().map(|value| value.as_deref()),
    );
    let precomputed = kernel_vec(&values, encoding);
    let bytes = strings.get_values_size();
    let dataset_sha256 = digest_input(&values);
    let rss_before_bytes = linux_rss_bytes("VmRSS:");

    let operation = || match mode.as_str() {
        "kernel_vec" => BenchmarkOutput::Vector(kernel_vec(black_box(&values), encoding)),
        "column_vec" => BenchmarkOutput::Vector(column_vec(black_box(&strings), encoding)),
        "column_arrow" => BenchmarkOutput::Arrow(column_arrow(black_box(&strings), encoding)),
        "output_only" => {
            BenchmarkOutput::Arrow(arrow_output(black_box(precomputed.iter().copied())))
        }
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
    let (counts_sha256, total_tokens) = match result.expect("at least one repeat") {
        BenchmarkOutput::Vector(values) => digest_counts(values),
        BenchmarkOutput::Arrow(values) => digest_counts(&values),
    };
    let median_seconds = median(&samples);
    let input_mib_per_second = (mode != "output_only").then(|| {
        f64::from(u32::try_from(bytes).expect("corpus bytes fit u32"))
            / median_seconds
            / (1024.0 * 1024.0)
    });
    let report = json!({
        "schema_version": 1,
        "mode": mode,
        "tokenizer": tokenizer,
        "rows": ROWS,
        "null_rows": ROWS / 100,
        "unique_values": ROWS - ROWS / 100,
        "bytes": bytes,
        "dataset_sha256": dataset_sha256,
        "counts_sha256": counts_sha256,
        "total_tokens": total_tokens,
        "samples_seconds": samples,
        "median_seconds": median_seconds,
        "input_mib_per_second": input_mib_per_second,
        "rss_before_bytes": rss_before_bytes,
        "rss_after_bytes": rss_after_bytes,
        "peak_rss_bytes": peak_rss_bytes,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize report")
    );
}

fn main() {
    if env::var_os(MODE_ENV).is_none() {
        eprintln!("skipping component_overhead benchmark: use the Python driver");
        return;
    }
    run();
}
