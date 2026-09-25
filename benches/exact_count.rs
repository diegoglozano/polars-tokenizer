use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use polars_tokenizer::{count_cl100k, count_o200k};

fn corpus(target_bytes: usize) -> String {
    const SAMPLE: &str = "The quick brown fox jumps over 13 lazy dogs. 你好👋\n";
    let mut text = String::with_capacity(target_bytes);
    for ch in SAMPLE.chars().cycle() {
        if text.len() + ch.len_utf8() > target_bytes {
            break;
        }
        text.push(ch);
    }
    text.extend(std::iter::repeat_n('x', target_bytes - text.len()));
    assert_eq!(text.len(), target_bytes);
    text
}

fn benchmark_exact_count(c: &mut Criterion) {
    black_box(count_o200k("warmup"));
    black_box(count_cl100k("warmup"));

    for (name, count) in [
        ("o200k_base_count", count_o200k as fn(&str) -> usize),
        ("cl100k_base_count", count_cl100k as fn(&str) -> usize),
    ] {
        let mut group = c.benchmark_group(name);
        for bytes in [32, 256, 4 * 1024, 128 * 1024] {
            let text = corpus(bytes);
            group.throughput(Throughput::Bytes(text.len() as u64));
            group.bench_with_input(BenchmarkId::from_parameter(bytes), &text, |b, text| {
                b.iter(|| count(black_box(text)));
            });
        }
        group.finish();
    }
}

fn benchmark_core_count_vs_encode(c: &mut Criterion) {
    for tokenizer in ["o200k_base", "cl100k_base"] {
        let encoding = tiktoken::get_encoding(tokenizer).expect("vocabulary must be compiled in");
        let mut group = c.benchmark_group(format!("{tokenizer}_core_count_vs_encode"));
        for bytes in [32, 256, 4 * 1024, 128 * 1024] {
            let text = corpus(bytes);
            assert_eq!(encoding.count(&text), encoding.encode(&text).len());
            group.throughput(Throughput::Bytes(text.len() as u64));
            group.bench_with_input(BenchmarkId::new("count", bytes), &text, |b, text| {
                b.iter(|| black_box(encoding.count(black_box(text))));
            });
            group.bench_with_input(BenchmarkId::new("encode", bytes), &text, |b, text| {
                b.iter(|| black_box(encoding.encode(black_box(text))));
            });
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    benchmark_exact_count,
    benchmark_core_count_vs_encode
);
criterion_main!(benches);
