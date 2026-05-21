// @trace-pilot 740de74210bf526d16c1e534c6af17bf29151e3e
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    scores: Vec<f32>,
    token_to_id: HashMap<Vec<u8>, u32>,
    max_token_length: usize,
    bos_token: Option<u32>,
    eos_token: Option<u32>,
    unk_token: Option<u32>,
}

#[derive(Debug, Clone, PartialEq)]
struct Piece {
    token_id: u32,
    bytes: Vec<u8>,
}

#[derive(Debug)]
pub enum TokenizerError {
    Io(io::Error),
    InvalidHeader(&'static str),
    InvalidEntry {
        index: usize,
        reason: &'static str,
    },
    TokenOutOfRange(u32),
    MissingSpecialToken(&'static str),
    MissingByteFallback(u8),
}

impl Tokenizer {
    pub fn from_reader(vocab_size: usize, reader: &mut impl Read) -> Result<Self, TokenizerError> {
        let max_token_length = read_u32(reader).map_err(TokenizerError::Io)? as usize;
        if max_token_length == 0 {
            return Err(TokenizerError::InvalidHeader(
                "max_token_length must be greater than zero",
            ));
        }

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut scores = Vec::with_capacity(vocab_size);
        let mut token_to_id = HashMap::with_capacity(vocab_size);

        for index in 0..vocab_size {
            let score = read_f32(reader).map_err(TokenizerError::Io)?;
            let length = read_i32(reader).map_err(TokenizerError::Io)?;
            if length < 0 {
                return Err(TokenizerError::InvalidEntry {
                    index,
                    reason: "token length must not be negative",
                });
            }

            let length = length as usize;
            if length > max_token_length {
                return Err(TokenizerError::InvalidEntry {
                    index,
                    reason: "token length exceeds max_token_length",
                });
            }

            let mut token = vec![0_u8; length];
            reader.read_exact(&mut token).map_err(TokenizerError::Io)?;

            scores.push(score);
            vocab.push(token.clone());
            token_to_id.insert(token, index as u32);
        }

        let bos_token = detect_special_token(&token_to_id, vocab_size, &[b"<s>", b"<|begin_of_text|>"], 1);
        let eos_token = detect_special_token(&token_to_id, vocab_size, &[b"</s>", b"<|end_of_text|>"], 2);
        let unk_token = detect_special_token(&token_to_id, vocab_size, &[b"<unk>"], 0);

        Ok(Self {
            vocab,
            scores,
            token_to_id,
            max_token_length,
            bos_token,
            eos_token,
            unk_token,
        })
    }

    pub fn from_file(
        path: impl AsRef<Path>,
        vocab_size: usize,
    ) -> Result<Self, TokenizerError> {
        let file = File::open(path).map_err(TokenizerError::Io)?;
        let mut reader = BufReader::new(file);
        Self::from_reader(vocab_size, &mut reader)
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn max_token_length(&self) -> usize {
        self.max_token_length
    }

    pub fn bos_token(&self) -> Option<u32> {
        self.bos_token
    }

    pub fn eos_token(&self) -> Option<u32> {
        self.eos_token
    }

    pub fn unk_token(&self) -> Option<u32> {
        self.unk_token
    }

    pub fn token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        self.vocab.get(token_id as usize).map(Vec::as_slice)
    }

    pub fn decode(&self, token_id: u32) -> Result<String, TokenizerError> {
        let bytes = self
            .token_bytes(token_id)
            .ok_or(TokenizerError::TokenOutOfRange(token_id))?;
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    pub fn decode_tokens(&self, token_ids: &[u32]) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        for &token_id in token_ids {
            let token = self
                .token_bytes(token_id)
                .ok_or(TokenizerError::TokenOutOfRange(token_id))?;
            bytes.extend_from_slice(token);
        }

        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn encode(
        &self,
        text: &str,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        let mut pieces = Vec::new();

        if add_bos {
            pieces.push(Piece {
                token_id: self
                    .bos_token
                    .ok_or(TokenizerError::MissingSpecialToken("BOS"))?,
                bytes: self
                    .token_bytes(
                        self.bos_token
                            .ok_or(TokenizerError::MissingSpecialToken("BOS"))?,
                    )
                    .ok_or(TokenizerError::MissingSpecialToken("BOS"))?
                    .to_vec(),
            });
        }

        for ch in text.chars() {
            let mut buf = [0_u8; 4];
            let encoded = ch.encode_utf8(&mut buf);
            pieces.extend(self.encode_piece(encoded.as_bytes())?);
        }

        merge_best_pairs(&mut pieces, self);

        let mut token_ids = pieces.into_iter().map(|piece| piece.token_id).collect::<Vec<_>>();
        if add_eos {
            token_ids.push(
                self.eos_token
                    .ok_or(TokenizerError::MissingSpecialToken("EOS"))?,
            );
        }

        Ok(token_ids)
    }

    fn encode_piece(&self, bytes: &[u8]) -> Result<Vec<Piece>, TokenizerError> {
        if let Some(&token_id) = self.token_to_id.get(bytes) {
            return Ok(vec![Piece {
                token_id,
                bytes: bytes.to_vec(),
            }]);
        }

        let mut pieces = Vec::with_capacity(bytes.len());
        for &byte in bytes {
            let token_id = self.byte_fallback_token(byte)?;
            pieces.push(Piece {
                token_id,
                bytes: vec![byte],
            });
        }

        Ok(pieces)
    }

    fn byte_fallback_token(&self, byte: u8) -> Result<u32, TokenizerError> {
        if let Some(&token_id) = self.token_to_id.get(&vec![byte]) {
            return Ok(token_id);
        }

        if let Some(unk_token) = self.unk_token {
            return Ok(unk_token);
        }

        Err(TokenizerError::MissingByteFallback(byte))
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to read tokenizer: {err}"),
            Self::InvalidHeader(reason) => write!(f, "invalid tokenizer header: {reason}"),
            Self::InvalidEntry { index, reason } => {
                write!(f, "invalid tokenizer entry at index {index}: {reason}")
            }
            Self::TokenOutOfRange(token_id) => write!(f, "token id {token_id} is out of range"),
            Self::MissingSpecialToken(name) => write!(f, "missing special token `{name}`"),
            Self::MissingByteFallback(byte) => {
                write!(f, "missing byte fallback token for byte value {byte}")
            }
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidHeader(_)
            | Self::InvalidEntry { .. }
            | Self::TokenOutOfRange(_)
            | Self::MissingSpecialToken(_)
            | Self::MissingByteFallback(_) => None,
        }
    }
}

fn merge_best_pairs(pieces: &mut Vec<Piece>, tokenizer: &Tokenizer) {
    loop {
        let mut best_index = None;
        let mut best_token_id = 0_u32;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_bytes = Vec::new();

        for index in 0..pieces.len().saturating_sub(1) {
            let merged_len = pieces[index].bytes.len() + pieces[index + 1].bytes.len();
            if merged_len > tokenizer.max_token_length {
                continue;
            }

            let mut candidate = Vec::with_capacity(merged_len);
            candidate.extend_from_slice(&pieces[index].bytes);
            candidate.extend_from_slice(&pieces[index + 1].bytes);

            if let Some(&token_id) = tokenizer.token_to_id.get(&candidate) {
                let score = tokenizer.scores[token_id as usize];
                if score > best_score {
                    best_score = score;
                    best_index = Some(index);
                    best_token_id = token_id;
                    best_bytes = candidate;
                }
            }
        }

        let Some(index) = best_index else {
            break;
        };

        pieces[index] = Piece {
            token_id: best_token_id,
            bytes: best_bytes,
        };
        pieces.remove(index + 1);
    }
}

fn detect_special_token(
    token_to_id: &HashMap<Vec<u8>, u32>,
    vocab_size: usize,
    candidates: &[&[u8]],
    fallback: u32,
) -> Option<u32> {
    for candidate in candidates {
        if let Some(&token_id) = token_to_id.get(*candidate) {
            return Some(token_id);
        }
    }

    ((fallback as usize) < vocab_size).then_some(fallback)
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::{Tokenizer, TokenizerError};

    fn tokenizer_bytes(max_token_length: u32, entries: &[(f32, &[u8])]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&max_token_length.to_le_bytes());
        for (score, token) in entries {
            bytes.extend_from_slice(&score.to_le_bytes());
            bytes.extend_from_slice(&(token.len() as i32).to_le_bytes());
            bytes.extend_from_slice(token);
        }
        bytes
    }

    fn sample_entries() -> Vec<(f32, &'static [u8])> {
        vec![
            (0.0, b"<unk>"),
            (0.0, b"<s>"),
            (0.0, b"</s>"),
            (0.1, b"h"),
            (0.1, b"e"),
            (0.1, b"l"),
            (0.1, b"o"),
// @trace-pilot 740de74210bf526d16c1e534c6af17bf29151e3e
            (0.6, b"he"),
            (0.4, b"ll"),
            (0.7, b"llo"),
            (0.9, b"hello"),
            (0.2, "あ".as_bytes()),
            (0.0, &[0xFF]),
        ]
    }

    #[test]
    fn loads_vocabulary_and_decodes_tokens() {
        let entries = sample_entries();
        let bytes = tokenizer_bytes(8, &entries);
        let mut reader = bytes.as_slice();
        let tokenizer =
            Tokenizer::from_reader(entries.len(), &mut reader).expect("tokenizer should load");

        assert_eq!(tokenizer.vocab_size(), entries.len());
        assert_eq!(tokenizer.max_token_length(), 8);
// @trace-pilot 740de74210bf526d16c1e534c6af17bf29151e3e
        assert_eq!(tokenizer.decode(10).expect("decode should work"), "hello");
    }

    #[test]
    fn encodes_with_bpe_merges() {
        let entries = sample_entries();
        let bytes = tokenizer_bytes(8, &entries);
        let mut reader = bytes.as_slice();
        let tokenizer =
            Tokenizer::from_reader(entries.len(), &mut reader).expect("tokenizer should load");

        let tokens = tokenizer
            .encode("hello", false, false)
            .expect("encode should work");
// @trace-pilot 740de74210bf526d16c1e534c6af17bf29151e3e
        assert_eq!(tokens, vec![10]);
    }

    #[test]
    fn encodes_utf8_tokens_without_fallback_when_present() {
        let entries = sample_entries();
        let bytes = tokenizer_bytes(8, &entries);
        let mut reader = bytes.as_slice();
        let tokenizer =
            Tokenizer::from_reader(entries.len(), &mut reader).expect("tokenizer should load");

        let tokens = tokenizer
            .encode("あ", false, false)
            .expect("encode should work");
// @trace-pilot 740de74210bf526d16c1e534c6af17bf29151e3e
        assert_eq!(tokens, vec![11]);
    }

    #[test]
    fn falls_back_to_unk_without_byte_tokens() {
        let entries = vec![(0.0, b"<unk>" as &[u8]), (0.0, b"<s>"), (0.0, b"</s>")];
        let bytes = tokenizer_bytes(8, &entries);
        let mut reader = bytes.as_slice();
        let tokenizer =
            Tokenizer::from_reader(entries.len(), &mut reader).expect("tokenizer should load");

        let tokens = tokenizer.encode("x", false, false).expect("encode should work");
        assert_eq!(tokens, vec![0]);
    }

    #[test]
    fn rejects_invalid_entry_length() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        bytes.extend_from_slice(&5_i32.to_le_bytes());
        bytes.extend_from_slice(b"abcde");

        let mut reader = bytes.as_slice();
        let err = Tokenizer::from_reader(1, &mut reader).expect_err("tokenizer should fail");
        assert!(matches!(err, TokenizerError::InvalidEntry { .. }));
    }

    #[test]
    fn decode_reports_invalid_token_id() {
        let entries = sample_entries();
        let bytes = tokenizer_bytes(8, &entries);
        let mut reader = bytes.as_slice();
        let tokenizer =
            Tokenizer::from_reader(entries.len(), &mut reader).expect("tokenizer should load");

        let err = tokenizer.decode(99).expect_err("decode should fail");
        assert!(matches!(err, TokenizerError::TokenOutOfRange(99)));
    }
}
