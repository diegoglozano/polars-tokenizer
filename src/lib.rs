//! Native count-only token expressions for Polars.

use polars::prelude::*;
use pyo3_polars::PolarsAllocator;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[global_allocator]
static ALLOCATOR: PolarsAllocator = PolarsAllocator::new();

const O200K_BASE: &str = "o200k_base";

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

#[polars_expr(output_type=UInt32)]
#[allow(clippy::needless_pass_by_value)] // Required by the plugin macro ABI.
fn token_count(inputs: &[Series], kwargs: CountKwargs) -> PolarsResult<Series> {
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
    let mut output =
        PrimitiveChunkedBuilder::<UInt32Type>::new(strings.name().clone(), strings.len());

    // Iteration borrows values from Polars' StringView buffers. The only
    // per-Series allocation here is the final values/validity output.
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

    Ok(output.finish().into_series())
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
}
