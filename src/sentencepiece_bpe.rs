//! Count-only SentencePiece BPE support for the pinned Gemma 3 tokenizer.
//!
//! This module intentionally implements only the model configuration used by
//! Google's pinned Gemma 3 artifact. Unsupported SentencePiece features are
//! rejected at load time instead of being approximated.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::error::Error;
use std::fmt::{self, Display, Formatter};

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, Anchored, Input, MatchKind, StartKind};
use rustc_hash::{FxBuildHasher, FxHashMap};
use sentencepiece_model::{ModelType, SentencePieceModel, Type};

const SPACE_SYMBOL: &str = "▁";

#[derive(Clone, Copy, Debug)]
struct PieceInfo {
    score: f32,
}

#[derive(Clone, Copy, Debug)]
struct Symbol {
    start: usize,
    len: usize,
    prev: Option<usize>,
    next: Option<usize>,
    frozen: bool,
}

#[derive(Clone, Copy, Debug)]
struct Pair {
    score: f32,
    left: usize,
    right: usize,
    size: usize,
}

impl PartialEq for Pair {
    fn eq(&self, other: &Self) -> bool {
        self.score.to_bits() == other.score.to_bits() && self.left == other.left
    }
}

impl Eq for Pair {}

impl PartialOrd for Pair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Pair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            // SentencePiece resolves equal scores toward the earlier pair.
            .then_with(|| other.left.cmp(&self.left))
    }
}

#[derive(Debug)]
pub enum ModelError {
    Decode(String),
    Unsupported(String),
    Invalid(String),
}

impl Display for ModelError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Decode(message) => write!(formatter, "invalid SentencePiece protobuf: {message}"),
            Self::Unsupported(message) => {
                write!(
                    formatter,
                    "unsupported SentencePiece configuration: {message}"
                )
            }
            Self::Invalid(message) => write!(formatter, "invalid SentencePiece model: {message}"),
        }
    }
}

impl Error for ModelError {}

/// A count-only `SentencePiece` BPE tokenizer for the pinned Gemma 3 model.
///
/// It never creates token IDs. Unknown final pieces contribute one token per
/// source byte, matching `SentencePiece` byte fallback.
pub struct SentencePieceBpe {
    pieces: FxHashMap<Box<str>, PieceInfo>,
    user_defined: Option<AhoCorasick>,
    user_defined_starts: [bool; 256],
}

impl SentencePieceBpe {
    pub fn from_model_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        let model = SentencePieceModel::from_slice(bytes)
            .map_err(|error| ModelError::Decode(error.to_string()))?;
        Self::validate_configuration(&model)?;

        let mut pieces = FxHashMap::with_capacity_and_hasher(model.pieces().len(), FxBuildHasher);
        let mut user_defined = Vec::new();
        let mut seen = HashSet::with_capacity(model.pieces().len());
        let mut unknown_count = 0_usize;
        let mut byte_values = [false; 256];

        for piece in model.pieces() {
            let text = piece.piece();
            if text.is_empty() {
                return Err(ModelError::Invalid("piece text must not be empty".into()));
            }
            if text.contains('\0') {
                return Err(ModelError::Invalid(
                    "piece text must not contain a NUL byte".into(),
                ));
            }
            if !seen.insert(text) {
                return Err(ModelError::Invalid(format!("duplicate piece {text:?}")));
            }

            let piece_type = piece.r#type();
            match piece_type {
                Type::Normal | Type::UserDefined => {
                    let score = piece.score();
                    if !score.is_finite() {
                        return Err(ModelError::Invalid(format!(
                            "piece {text:?} has a non-finite score"
                        )));
                    }
                    if piece_type == Type::UserDefined {
                        user_defined.push(text.to_owned());
                    }
                    pieces.insert(text.into(), PieceInfo { score });
                }
                Type::Unknown => unknown_count += 1,
                Type::Byte => {
                    let byte = parse_byte_piece(text).ok_or_else(|| {
                        ModelError::Invalid(format!("malformed byte piece {text:?}"))
                    })?;
                    if std::mem::replace(&mut byte_values[usize::from(byte)], true) {
                        return Err(ModelError::Invalid(format!(
                            "duplicate byte piece for 0x{byte:02X}"
                        )));
                    }
                }
                Type::Control => {}
                Type::Unused => {
                    return Err(ModelError::Unsupported(
                        "UNUSED-piece resegmentation is not implemented".into(),
                    ));
                }
            }
        }

        if unknown_count != 1 {
            return Err(ModelError::Invalid(format!(
                "expected one UNKNOWN piece, found {unknown_count}"
            )));
        }
        let byte_count = byte_values.iter().filter(|present| **present).count();
        if byte_count != byte_values.len() {
            return Err(ModelError::Invalid(format!(
                "byte fallback requires 256 byte pieces, found {byte_count}"
            )));
        }

        let (user_defined, user_defined_starts) = build_prefix_matcher(&user_defined)?;
        Ok(Self {
            pieces,
            user_defined,
            user_defined_starts,
        })
    }

    fn validate_configuration(model: &SentencePieceModel) -> Result<(), ModelError> {
        let trainer = model
            .trainer()
            .ok_or_else(|| ModelError::Invalid("missing TrainerSpec".into()))?;
        if trainer.model_type() != ModelType::Bpe {
            return Err(ModelError::Unsupported(format!(
                "model type {:?}; expected BPE",
                trainer.model_type()
            )));
        }
        if !trainer.byte_fallback() {
            return Err(ModelError::Unsupported(
                "byte fallback must be enabled".into(),
            ));
        }
        if trainer.treat_whitespace_as_suffix() {
            return Err(ModelError::Unsupported(
                "whitespace-as-suffix normalization".into(),
            ));
        }
        if !trainer.pretokenization_delimiter().is_empty() {
            return Err(ModelError::Unsupported(
                "pre-tokenization delimiters".into(),
            ));
        }

        let normalizer = model
            .normalizer()
            .ok_or_else(|| ModelError::Invalid("missing NormalizerSpec".into()))?;
        if normalizer.name() != "identity" || !normalizer.precompiled_charsmap().is_empty() {
            return Err(ModelError::Unsupported(
                "only identity normalization without a compiled character map is supported".into(),
            ));
        }
        if normalizer.add_dummy_prefix() {
            return Err(ModelError::Unsupported("dummy-prefix insertion".into()));
        }
        if normalizer.remove_extra_whitespaces() {
            return Err(ModelError::Unsupported(
                "whitespace removal or collapsing".into(),
            ));
        }
        if !normalizer.escape_whitespaces() {
            return Err(ModelError::Unsupported(
                "models that do not escape ASCII spaces".into(),
            ));
        }
        if model.denormalizer().is_some() {
            return Err(ModelError::Unsupported("denormalization rules".into()));
        }
        Ok(())
    }

    /// Returns the exact number of tokens without constructing token IDs.
    #[must_use]
    pub fn count(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }

        let (normalized, mut symbols) = self.initial_symbols(text);
        let mut agenda = BinaryHeap::new();
        for left in 0..symbols.len().saturating_sub(1) {
            self.maybe_add_pair(&normalized, &symbols, left, left + 1, &mut agenda);
        }

        while let Some(pair) = agenda.pop() {
            if symbols[pair.left].len == 0
                || symbols[pair.right].len == 0
                || symbols[pair.left].len + symbols[pair.right].len != pair.size
            {
                continue;
            }

            let previous = symbols[pair.left].prev;
            let following = symbols[pair.right].next;
            let right_len = symbols[pair.right].len;
            symbols[pair.left].len += right_len;
            symbols[pair.left].next = following;
            symbols[pair.right].len = 0;

            if let Some(next) = following {
                symbols[next].prev = Some(pair.left);
            }
            if let Some(prev) = previous {
                self.maybe_add_pair(&normalized, &symbols, prev, pair.left, &mut agenda);
            }
            if let Some(next) = following {
                self.maybe_add_pair(&normalized, &symbols, pair.left, next, &mut agenda);
            }
        }

        let mut count = 0_usize;
        let mut current = Some(0_usize);
        while let Some(index) = current {
            let symbol = symbols[index];
            if symbol.len > 0 {
                let piece = &normalized[symbol.start..symbol.start + symbol.len];
                count += if self.pieces.contains_key(piece) {
                    1
                } else {
                    // SentencePiece emits one BYTE token for every UTF-8 byte
                    // in an unknown final symbol.
                    piece.len()
                };
            }
            current = symbol.next;
        }
        count
    }

    fn initial_symbols(&self, text: &str) -> (String, Vec<Symbol>) {
        let mut normalized = String::with_capacity(text.len());
        let mut symbols = Vec::with_capacity(text.len().min(4_096));
        let mut input_offset = 0_usize;

        while input_offset < text.len() {
            let remaining = &text[input_offset..];
            let first = usize::from(remaining.as_bytes()[0]);
            let user_defined_len = self.user_defined_starts[first]
                .then_some(self.user_defined.as_ref())
                .flatten()
                .and_then(|matcher| {
                    matcher
                        .find(Input::new(remaining).anchored(Anchored::Yes))
                        .map(|matched| matched.end())
                });

            let (input_len, frozen) = user_defined_len.map_or_else(
                || {
                    (
                        remaining
                            .chars()
                            .next()
                            .expect("non-empty suffix")
                            .len_utf8(),
                        false,
                    )
                },
                |len| (len, true),
            );
            let source = &remaining[..input_len];
            let start = normalized.len();
            if !frozen && source == " " {
                normalized.push_str(SPACE_SYMBOL);
            } else {
                normalized.push_str(source);
            }
            let len = normalized.len() - start;
            let index = symbols.len();
            symbols.push(Symbol {
                start,
                len,
                prev: index.checked_sub(1),
                next: None,
                frozen,
            });
            if index > 0 {
                symbols[index - 1].next = Some(index);
            }
            input_offset += input_len;
        }

        (normalized, symbols)
    }

    fn maybe_add_pair(
        &self,
        normalized: &str,
        symbols: &[Symbol],
        left: usize,
        right: usize,
        agenda: &mut BinaryHeap<Pair>,
    ) {
        if symbols[left].frozen || symbols[right].frozen {
            return;
        }
        let size = symbols[left].len + symbols[right].len;
        let start = symbols[left].start;
        let merged = &normalized[start..start + size];
        if let Some(piece) = self.pieces.get(merged) {
            agenda.push(Pair {
                score: piece.score,
                left,
                right,
                size,
            });
        }
    }

    #[cfg(test)]
    fn for_test(normal: &[(&str, f32)], user_defined: &[(&str, f32)]) -> Self {
        let mut pieces = FxHashMap::default();
        for (text, score) in normal {
            pieces.insert(Box::<str>::from(*text), PieceInfo { score: *score });
        }
        let patterns: Vec<String> = user_defined
            .iter()
            .map(|(text, _)| (*text).to_owned())
            .collect();
        for (text, score) in user_defined {
            pieces.insert(Box::<str>::from(*text), PieceInfo { score: *score });
        }
        let (user_defined, user_defined_starts) =
            build_prefix_matcher(&patterns).expect("valid test patterns");
        Self {
            pieces,
            user_defined,
            user_defined_starts,
        }
    }
}

fn build_prefix_matcher(
    patterns: &[String],
) -> Result<(Option<AhoCorasick>, [bool; 256]), ModelError> {
    let mut starts = [false; 256];
    if patterns.is_empty() {
        return Ok((None, starts));
    }
    for pattern in patterns {
        starts[usize::from(pattern.as_bytes()[0])] = true;
    }
    AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .start_kind(StartKind::Anchored)
        .build(patterns)
        .map(Some)
        .map(|matcher| (matcher, starts))
        .map_err(|error| ModelError::Invalid(format!("invalid user-defined symbols: {error}")))
}

fn parse_byte_piece(piece: &str) -> Option<u8> {
    let hexadecimal = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if hexadecimal.len() != 2 {
        return None;
    }
    u8::from_str_radix(hexadecimal, 16).ok()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::io::{BufRead, BufReader};

    use super::{SentencePieceBpe, parse_byte_piece};

    #[test]
    fn parses_byte_pieces_strictly() {
        assert_eq!(parse_byte_piece("<0x00>"), Some(0));
        assert_eq!(parse_byte_piece("<0xFF>"), Some(255));
        assert_eq!(parse_byte_piece("<0xff>"), Some(255));
        assert_eq!(parse_byte_piece("<0xF>"), None);
        assert_eq!(parse_byte_piece("0xFF"), None);
    }

    #[test]
    fn merges_and_escapes_spaces() {
        let tokenizer = SentencePieceBpe::for_test(
            &[
                ("h", 0.0),
                ("i", 0.0),
                ("▁", 0.0),
                ("hi", 1.0),
                ("▁hi", 2.0),
            ],
            &[],
        );
        assert_eq!(tokenizer.count("hi"), 1);
        assert_eq!(tokenizer.count(" hi"), 1);
        assert_eq!(tokenizer.count(""), 0);
    }

    #[test]
    fn unknowns_fall_back_to_each_utf8_byte() {
        let tokenizer = SentencePieceBpe::for_test(&[], &[]);
        assert_eq!(tokenizer.count("a"), 1);
        assert_eq!(tokenizer.count("é"), 2);
        assert_eq!(tokenizer.count("🦀"), 4);
        assert_eq!(tokenizer.count("\0"), 1);
    }

    #[test]
    fn longest_user_defined_prefix_is_frozen() {
        let tokenizer = SentencePieceBpe::for_test(
            &[("x", 0.0), ("<tag>x", 100.0)],
            &[("<t", 0.0), ("<tag>", 0.0)],
        );
        assert_eq!(tokenizer.count("<tag>x"), 2);
        assert_eq!(tokenizer.count("<t"), 1);
    }

    /// Run explicitly with the official pinned artifact:
    ///
    /// `POLARS_TOKENIZER_GEMMA3_MODEL=/path/to/model cargo test \
    /// sentencepiece_bpe::tests::matches_pinned_gemma3_reference -- --ignored`
    #[test]
    #[ignore = "requires Google's pinned Gemma 3 SentencePiece artifact"]
    fn matches_pinned_gemma3_reference() {
        let path = std::env::var("POLARS_TOKENIZER_GEMMA3_MODEL")
            .expect("POLARS_TOKENIZER_GEMMA3_MODEL must point to the pinned model");
        let bytes = fs::read(path).expect("read pinned Gemma 3 model");
        let tokenizer = SentencePieceBpe::from_model_bytes(&bytes).expect("load pinned model");

        // Counts generated by google.genai.local_tokenizer.LocalTokenizer for
        // the SHA-256-pinned gemma3 model. Keep cases focused on normalization,
        // Unicode, byte fallback, and user-defined symbols.
        let cases = [
            ("", 0),
            ("hello world", 2),
            (" hello", 1),
            ("hello  world", 3),
            ("line one\r\nline two", 6),
            ("null\0byte", 3),
            ("e\u{301}", 2),
            ("é", 1),
            ("你好，世界", 3),
            ("مرحبا بالعالم", 4),
            ("🦀🚀", 2),
            ("<start_of_turn>user", 2),
        ];
        for (text, expected) in cases {
            assert_eq!(tokenizer.count(text), expected, "input {text:?}");
        }
    }

    /// Checks a large JSON-lines oracle without committing Google's model or
    /// generated token IDs. Each line is `[text, expected_count]`.
    #[test]
    #[ignore = "requires the pinned model and a generated reference oracle"]
    fn matches_gemma3_reference_oracle() {
        let model_path = std::env::var("POLARS_TOKENIZER_GEMMA3_MODEL")
            .expect("POLARS_TOKENIZER_GEMMA3_MODEL must point to the pinned model");
        let oracle_path = std::env::var("POLARS_TOKENIZER_GEMMA3_ORACLE")
            .expect("POLARS_TOKENIZER_GEMMA3_ORACLE must point to JSON-lines counts");
        let bytes = fs::read(model_path).expect("read pinned Gemma 3 model");
        let tokenizer = SentencePieceBpe::from_model_bytes(&bytes).expect("load pinned model");
        let oracle = fs::File::open(oracle_path).expect("open reference oracle");

        let mut cases = 0_usize;
        for (line_number, line) in BufReader::new(oracle).lines().enumerate() {
            let line = line.expect("read reference oracle line");
            let (text, expected): (String, usize) = serde_json::from_str(&line)
                .unwrap_or_else(|error| panic!("invalid oracle line {}: {error}", line_number + 1));
            assert_eq!(
                tokenizer.count(&text),
                expected,
                "oracle line {} input {text:?}",
                line_number + 1
            );
            cases += 1;
        }
        assert!(cases > 0, "reference oracle must not be empty");
    }
}
