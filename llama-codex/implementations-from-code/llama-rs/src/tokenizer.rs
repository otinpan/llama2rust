use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

#[derive(Debug, Clone)]
pub struct Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    sorted_vocab: HashMap<String, usize>,
    max_token_length: usize,
}

impl Tokenizer {
    pub fn from_file(path: &Path, vocab_size: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::from_reader(&mut reader, vocab_size)
    }

    pub(crate) fn from_reader<R: Read>(reader: &mut R, vocab_size: usize) -> io::Result<Self> {
        let max_token_length = read_u32(reader)? as usize;
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut vocab_scores = Vec::with_capacity(vocab_size);

        for _ in 0..vocab_size {
            vocab_scores.push(read_f32(reader)?);
            let len = read_u32(reader)? as usize;
            let mut bytes = vec![0_u8; len];
            reader.read_exact(&mut bytes)?;
            let piece = String::from_utf8(bytes).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid utf-8 in tokenizer vocab: {error}"),
                )
            })?;
            vocab.push(piece);
        }

        let sorted_vocab = vocab
            .iter()
            .cloned()
            .enumerate()
            .map(|(id, piece)| (piece, id))
            .collect();

        Ok(Self {
            vocab,
            vocab_scores,
            sorted_vocab,
            max_token_length,
        })
    }

    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> io::Result<Vec<usize>> {
        let mut tokens = Vec::with_capacity(text.len() + 3);

        if bos {
            tokens.push(1);
        }

        if !text.is_empty() {
            let dummy_prefix = self
                .lookup(" ")
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing dummy prefix token"))?;
            tokens.push(dummy_prefix);
        }

        let bytes = text.as_bytes();
        let mut codepoint = Vec::with_capacity(4);
        for (index, &byte) in bytes.iter().enumerate() {
            if (byte & 0xC0) != 0x80 {
                codepoint.clear();
            }
            codepoint.push(byte);

            let next_is_continuation = bytes
                .get(index + 1)
                .is_some_and(|next| (next & 0xC0) == 0x80);
            if next_is_continuation && codepoint.len() < 4 {
                continue;
            }

            if let Ok(piece) = std::str::from_utf8(&codepoint) {
                if let Some(id) = self.lookup(piece) {
                    tokens.push(id);
                } else {
                    for &raw_byte in &codepoint {
                        tokens.push(raw_byte as usize + 3);
                    }
                }
            } else {
                for &raw_byte in &codepoint {
                    tokens.push(raw_byte as usize + 3);
                }
            }
            codepoint.clear();
        }

        let mut merge_buffer = String::with_capacity(self.max_token_length * 2 + 3);
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_id = None;
            let mut best_index = None;

            for index in 0..tokens.len().saturating_sub(1) {
                merge_buffer.clear();
                merge_buffer.push_str(self.piece(tokens[index])?);
                merge_buffer.push_str(self.piece(tokens[index + 1])?);

                if let Some(id) = self.lookup(&merge_buffer) {
                    let score = self.vocab_scores[id];
                    if score > best_score {
                        best_score = score;
                        best_id = Some(id);
                        best_index = Some(index);
                    }
                }
            }

            let Some(index) = best_index else {
                break;
            };
            tokens[index] = best_id.expect("best_id set with best_index");
            tokens.remove(index + 1);
        }

        if eos {
            tokens.push(2);
        }

        Ok(tokens)
    }

    pub fn decode(&self, prev_token: usize, token: usize) -> io::Result<String> {
        let mut piece = self.piece(token)?.to_owned();

        if prev_token == 1 && piece.starts_with(' ') {
            piece.remove(0);
        }

        if let Some(byte) = parse_raw_byte_piece(&piece) {
            return Ok(String::from_utf8_lossy(&[byte]).into_owned());
        }

        Ok(piece)
    }

    pub fn safe_decode(&self, prev_token: usize, token: usize) -> io::Result<Option<String>> {
        let piece = self.decode(prev_token, token)?;
        if piece.is_empty() {
            return Ok(None);
        }
        if piece.len() == 1 {
            let byte = piece.as_bytes()[0];
            if !(byte.is_ascii_graphic() || byte.is_ascii_whitespace()) {
                return Ok(None);
            }
        }
        Ok(Some(piece))
    }

    fn lookup(&self, piece: &str) -> Option<usize> {
        self.sorted_vocab.get(piece).copied()
    }

    fn piece(&self, token: usize) -> io::Result<&str> {
        self.vocab.get(token).map(String::as_str).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("token out of range in tokenizer: {token}"),
            )
        })
    }
}

fn parse_raw_byte_piece(piece: &str) -> Option<u8> {
    if piece.len() != 6 || !piece.starts_with("<0x") || !piece.ends_with('>') {
        return None;
    }
    u8::from_str_radix(&piece[3..5], 16).ok()
}

fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_f32<R: Read>(reader: &mut R) -> io::Result<f32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::Tokenizer;

    #[test]
    fn loads_vocab_and_encodes_with_merges() {
        let mut cursor = Cursor::new(tokenizer_fixture_bytes());
        let tokenizer = Tokenizer::from_reader(&mut cursor, 8).unwrap();

        let tokens = tokenizer.encode("ab", true, false).unwrap();

        assert_eq!(tokens, vec![1, 3, 7]);
    }

    #[test]
    fn decodes_byte_piece_and_strips_bos_space() {
        let mut cursor = Cursor::new(tokenizer_fixture_bytes());
        let tokenizer = Tokenizer::from_reader(&mut cursor, 8).unwrap();

        assert_eq!(tokenizer.decode(1, 3).unwrap(), "");
        assert_eq!(tokenizer.decode(0, 6).unwrap(), "a");
    }

    fn tokenizer_fixture_bytes() -> Vec<u8> {
        let pieces = [
            ("<unk>", 0.0_f32),
            ("<s>", 0.0),
            ("</s>", 0.0),
            (" ", 0.0),
            ("a", 1.0),
            ("b", 1.0),
            ("<0x61>", 0.0),
            ("ab", 10.0),
        ];

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(8_u32).to_le_bytes());
        for (piece, score) in pieces {
            bytes.extend_from_slice(&score.to_le_bytes());
            bytes.extend_from_slice(&(piece.len() as u32).to_le_bytes());
            bytes.extend_from_slice(piece.as_bytes());
        }
        bytes
    }
}
