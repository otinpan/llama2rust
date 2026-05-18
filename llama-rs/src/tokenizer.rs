// @trace-pilot a753c602b9fc1a8a17f6019e74599aef2b3cc035
// Tokenizer

// @trace-pilot 6884d3a2b2265d5e4d213741554073e7f4dad7e5
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

#[derive(Debug)]
pub struct Tokenizer {
    pub vocab: Vec<String>,
    pub vocab_scores: Vec<f32>,
    pub max_token_length: usize,
}

impl Tokenizer {
    // @trace-pilot 6a5e681f77038ebba2c8b5ec7e66767ffb3503d3
    // void build_tokenizer
    pub fn new(path: impl AsRef<Path>, vocab_size: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        let max_token_length = read_u32(&mut reader)? as usize;
        let mut vocab = Vec::with_capacity(vocab_size);
        let mut vocab_scores = Vec::with_capacity(vocab_size);

        for _ in 0..vocab_size {
            let score = read_f32(&mut reader)?;
            let piece_len = read_u32(&mut reader)? as usize;
            let mut bytes = vec![0_u8; piece_len];
            reader.read_exact(&mut bytes)?;

            let piece = String::from_utf8(bytes).map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid tokenizer entry: {err}"),
                )
            })?;

            vocab_scores.push(score);
            vocab.push(piece);
        }

        Ok(Self {
            vocab,
            vocab_scores,
            max_token_length,
        })
    }
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
