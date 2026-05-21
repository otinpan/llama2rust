use std::io;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub seq_len: usize,
}

impl Config {
    pub const BYTE_SIZE: usize = 7 * std::mem::size_of::<i32>();

    pub fn new(
        dim: usize,
        hidden_dim: usize,
        n_layers: usize,
        n_heads: usize,
        n_kv_heads: usize,
        vocab_size: usize,
        seq_len: usize,
    ) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert!(hidden_dim > 0, "hidden_dim must be positive");
        assert!(n_layers > 0, "n_layers must be positive");
        assert!(n_heads > 0, "n_heads must be positive");
        assert!(n_kv_heads > 0, "n_kv_heads must be positive");
        assert!(vocab_size > 0, "vocab_size must be positive");
        assert!(seq_len > 0, "seq_len must be positive");
        assert!(dim.is_multiple_of(n_heads), "dim must be divisible by n_heads");
        assert!(
            n_heads.is_multiple_of(n_kv_heads),
            "n_heads must be divisible by n_kv_heads"
        );

        Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        }
    }

    pub fn head_size(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.head_size() * self.n_kv_heads
    }

    pub fn from_i32_array(values: [i32; 7]) -> io::Result<(Self, bool)> {
        let shared_weights = values[5] > 0;
        let vocab_size = values[5].unsigned_abs() as usize;

        let config = Self::new(
            positive_i32_to_usize(values[0], "dim")?,
            positive_i32_to_usize(values[1], "hidden_dim")?,
            positive_i32_to_usize(values[2], "n_layers")?,
            positive_i32_to_usize(values[3], "n_heads")?,
            positive_i32_to_usize(values[4], "n_kv_heads")?,
            positive_usize(vocab_size, "vocab_size")?,
            positive_i32_to_usize(values[6], "seq_len")?,
        );

        Ok((config, shared_weights))
    }

    pub fn from_bytes(bytes: [u8; Self::BYTE_SIZE]) -> io::Result<(Self, bool)> {
        let mut values = [0_i32; 7];
        for (index, chunk) in bytes.chunks_exact(4).enumerate() {
            values[index] = i32::from_le_bytes(chunk.try_into().expect("4-byte chunk"));
        }
        Self::from_i32_array(values)
    }
}

fn positive_i32_to_usize(value: i32, field: &str) -> io::Result<usize> {
    if value <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{field} must be positive, got {value}"),
        ));
    }
    Ok(value as usize)
}

fn positive_usize(value: usize, field: &str) -> io::Result<usize> {
    if value == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{field} must be positive, got 0"),
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn derives_head_sizes() {
        let config = Config::new(288, 768, 6, 6, 2, 32_000, 256);
        assert_eq!(config.head_size(), 48);
        assert_eq!(config.kv_dim(), 96);
    }

    #[test]
    fn decodes_header_and_shared_weight_flag() {
        let values = [288, 768, 6, 6, 2, -32_000, 256];
        let (config, shared_weights) = Config::from_i32_array(values).unwrap();

        assert!(!shared_weights);
        assert_eq!(config.vocab_size, 32_000);
        assert_eq!(config.kv_dim(), 96);
    }
}
