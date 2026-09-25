//! Experimental whole-value cache crossover benchmark, not a runtime policy.

use std::collections::{HashMap, VecDeque};
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tiktoken::CoreBpe;

const ROWS: usize = 100_000;
const TARGET_BYTES: usize = 128;
const CARDINALITIES: [(&str, usize); 6] = [
    ("0.01%", 10),
    ("0.1%", 100),
    ("1%", 1_000),
    ("10%", 10_000),
    ("50%", 50_000),
    ("100%", 100_000),
];
const CACHE_CAPACITIES: [usize; 2] = [256, 4_096];
const SAMPLE: &str = "The quick brown fox jumps over 13 lazy dogs. 你好👋 ";

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

fn corpus(unique: usize) -> Vec<String> {
    assert!((1..=ROWS).contains(&unique));
    let dictionary: Vec<String> = (0..unique).map(value).collect();
    assert!(dictionary.iter().all(|value| value.len() == TARGET_BYTES));
    let mut values: Vec<String> = (0..ROWS)
        .map(|index| dictionary[index % unique].clone())
        .collect();

    // Ensure every dictionary value occurs, then avoid a periodic cache pattern.
    let mut state = 42_usize;
    for index in (1..values.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let other = state % (index + 1);
        values.swap(index, other);
    }
    values
}

fn count_uncached(values: &[String], encoding: &CoreBpe) -> Vec<usize> {
    values.iter().map(|value| encoding.count(value)).collect()
}

struct CacheResult {
    counts: Vec<usize>,
    hits: usize,
    max_entries: usize,
}

fn count_cached(values: &[String], encoding: &CoreBpe, capacity: usize) -> CacheResult {
    assert!(capacity > 0);
    let mut cache: HashMap<&str, usize> = HashMap::new();
    let mut insertion_order = VecDeque::new();
    let mut counts = Vec::with_capacity(values.len());
    let mut hits = 0;
    let mut max_entries = 0;

    for value in values {
        if let Some(&count) = cache.get(value.as_str()) {
            hits += 1;
            counts.push(count);
            continue;
        }

        let count = encoding.count(value);
        if cache.len() == capacity {
            let oldest = insertion_order
                .pop_front()
                .expect("full cache has an entry");
            cache.remove(oldest);
        }
        cache.insert(value, count);
        insertion_order.push_back(value.as_str());
        max_entries = max_entries.max(cache.len());
        counts.push(count);
    }

    CacheResult {
        counts,
        hits,
        max_entries,
    }
}

fn benchmark_cache_crossover(c: &mut Criterion) {
    for tokenizer in ["o200k_base", "cl100k_base"] {
        let encoding = tiktoken::get_encoding(tokenizer).expect("vocabulary must be compiled in");
        for (cardinality, unique) in CARDINALITIES {
            let values = corpus(unique);
            let expected = count_uncached(&values, encoding);
            let total_bytes: usize = values.iter().map(String::len).sum();
            let mut group =
                c.benchmark_group(format!("whole_value_cache/{tokenizer}/{cardinality}"));
            group.throughput(Throughput::Bytes(
                u64::try_from(total_bytes).expect("corpus byte count fits u64"),
            ));
            group.bench_function(BenchmarkId::new("uncached", ROWS), |b| {
                b.iter(|| black_box(count_uncached(black_box(&values), encoding)));
            });

            for capacity in CACHE_CAPACITIES {
                let result = count_cached(&values, encoding, capacity);
                assert_eq!(result.counts, expected, "cache must preserve exact counts");
                assert!(result.max_entries <= capacity, "cache must stay bounded");
                if unique <= capacity {
                    assert_eq!(result.hits, ROWS - unique);
                }
                if unique == ROWS {
                    assert_eq!(result.hits, 0);
                }
                eprintln!(
                    "{tokenizer} cardinality={cardinality} capacity={capacity} hits={} max_entries={}",
                    result.hits, result.max_entries
                );
                group.bench_function(BenchmarkId::new("cache", capacity), |b| {
                    b.iter(|| black_box(count_cached(black_box(&values), encoding, capacity)));
                });
            }
            group.finish();
        }
    }
}

criterion_group!(benches, benchmark_cache_crossover);
criterion_main!(benches);
