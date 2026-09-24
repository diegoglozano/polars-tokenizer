use std::env;
use std::fs;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Duration;

use criterion::{Criterion, Throughput};
use polars_tokenizer::sentencepiece_bpe::SentencePieceBpe;

const MODEL_ENV: &str = "POLARS_TOKENIZER_GEMMA3_MODEL";
const ORACLE_ENV: &str = "POLARS_TOKENIZER_GEMMA3_ORACLE";

fn make_text(minimum_bytes: usize) -> String {
    const SAMPLE: &str = "customer requested refund 你好 🦀 https://example.com/a?q=1\n";
    let mut text = String::with_capacity(minimum_bytes + SAMPLE.len());
    while text.len() < minimum_bytes {
        text.push_str(SAMPLE);
    }
    text
}

fn load_oracle(path: &Path) -> (Vec<String>, usize) {
    let oracle = fs::File::open(path).expect("open reference oracle");
    let mut texts = Vec::new();
    let mut expected_total = 0_usize;
    for line in BufReader::new(oracle).lines() {
        let line = line.expect("read reference oracle line");
        let (text, expected): (String, usize) =
            serde_json::from_str(&line).expect("parse reference oracle line");
        texts.push(text);
        expected_total += expected;
    }
    (texts, expected_total)
}

fn main() {
    let Ok(model_path) = env::var(MODEL_ENV) else {
        eprintln!("skipping gemma3_count benchmark: set {MODEL_ENV} to the pinned model path");
        return;
    };
    let model_bytes = fs::read(model_path).expect("read pinned Gemma 3 model");
    let tokenizer = SentencePieceBpe::from_model_bytes(&model_bytes).expect("load Gemma 3 model");
    let mut criterion = Criterion::default()
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
        .configure_from_args();

    let mut initialization = criterion.benchmark_group("gemma3_model_initialization");
    initialization.sample_size(10);
    initialization.measurement_time(Duration::from_secs(6));
    initialization.throughput(Throughput::Bytes(model_bytes.len() as u64));
    initialization.bench_function("parse_validate_index", |bencher| {
        bencher.iter(|| {
            SentencePieceBpe::from_model_bytes(black_box(&model_bytes)).expect("load Gemma 3 model")
        });
    });
    initialization.finish();

    let mut count = criterion.benchmark_group("gemma3_count_only");
    count.sample_size(50);
    for minimum_bytes in [32, 256, 4_096, 131_072] {
        let text = make_text(minimum_bytes);
        count.throughput(Throughput::Bytes(text.len() as u64));
        count.bench_with_input(minimum_bytes.to_string(), &text, |bencher, text| {
            bencher.iter(|| tokenizer.count(black_box(text)));
        });
    }
    count.finish();

    if let Ok(oracle_path) = env::var(ORACLE_ENV) {
        let (texts, expected_total) = load_oracle(Path::new(&oracle_path));
        let actual_total: usize = texts.iter().map(|text| tokenizer.count(text)).sum();
        assert_eq!(actual_total, expected_total, "oracle counts must match");
        let bytes = texts.iter().map(String::len).sum::<usize>();
        let mut oracle = criterion.benchmark_group("gemma3_reference_oracle");
        oracle.sample_size(20);
        oracle.throughput(Throughput::Bytes(bytes as u64));
        oracle.bench_function("count_all", |bencher| {
            bencher.iter(|| {
                texts
                    .iter()
                    .map(|text| tokenizer.count(black_box(text)))
                    .sum::<usize>()
            });
        });
        oracle.finish();
    }

    criterion.final_summary();
}
