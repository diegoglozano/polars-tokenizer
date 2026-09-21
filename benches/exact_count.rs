use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use polars_tokenizer::count_o200k;

fn corpus(target_bytes: usize) -> String {
    const SAMPLE: &str = "The quick brown fox jumps over 13 lazy dogs. 你好👋\n";
    SAMPLE.repeat(target_bytes.div_ceil(SAMPLE.len()))
}

fn benchmark_exact_count(c: &mut Criterion) {
    black_box(count_o200k("warmup"));

    let mut group = c.benchmark_group("o200k_base_count");
    for bytes in [32, 256, 4 * 1024, 128 * 1024] {
        let text = corpus(bytes);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(bytes), &text, |b, text| {
            b.iter(|| count_o200k(black_box(text)));
        });
    }
    group.finish();
}

criterion_group!(benches, benchmark_exact_count);
criterion_main!(benches);
