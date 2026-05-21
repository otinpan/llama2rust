// @trace-pilot a8727e84ecdf804978f3e9cd2100ed5d43740b4f
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::config::Config;

#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    pub token_embedding_table: Vec<f32>,
    pub rms_att_weight: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub rms_ffn_weight: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
    pub rms_final_weight: Vec<f32>,
    pub freq_cis_real: Vec<f32>,
    pub freq_cis_imag: Vec<f32>,
    pub wcls: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightCounts {
    pub token_embedding_table: usize,
    pub rms_att_weight: usize,
    pub wq: usize,
    pub wk: usize,
    pub wv: usize,
    pub wo: usize,
    pub rms_ffn_weight: usize,
    pub w1: usize,
    pub w2: usize,
    pub w3: usize,
    pub rms_final_weight: usize,
    pub freq_cis_real: usize,
    pub freq_cis_imag: usize,
    pub wcls: usize,
}

#[derive(Debug)]
pub enum WeightsError {
    Io(io::Error),
    InvalidLayout(&'static str),
    TrailingBytes {
        expected: usize,
        actual: usize,
    },
    SizeOverflow(&'static str),
}

impl Weights {
    pub fn from_reader(config: &Config, reader: &mut impl Read) -> Result<Self, WeightsError> {
        let counts = WeightCounts::from_config(config)?;

        let token_embedding_table = read_f32_tensor(reader, counts.token_embedding_table)?;
        let rms_att_weight = read_f32_tensor(reader, counts.rms_att_weight)?;
        let wq = read_f32_tensor(reader, counts.wq)?;
        let wk = read_f32_tensor(reader, counts.wk)?;
        let wv = read_f32_tensor(reader, counts.wv)?;
        let wo = read_f32_tensor(reader, counts.wo)?;
        let rms_ffn_weight = read_f32_tensor(reader, counts.rms_ffn_weight)?;
        let w1 = read_f32_tensor(reader, counts.w1)?;
        let w2 = read_f32_tensor(reader, counts.w2)?;
        let w3 = read_f32_tensor(reader, counts.w3)?;
        let rms_final_weight = read_f32_tensor(reader, counts.rms_final_weight)?;
        let freq_cis_real = read_f32_tensor(reader, counts.freq_cis_real)?;
        let freq_cis_imag = read_f32_tensor(reader, counts.freq_cis_imag)?;

        let mut trailing = Vec::new();
        reader.read_to_end(&mut trailing).map_err(WeightsError::Io)?;
        let wcls = match trailing.len() {
            0 => None,
            bytes if bytes == counts.wcls * std::mem::size_of::<f32>() => {
                Some(bytes_to_f32_vec(trailing)?)
            }
            actual => {
                return Err(WeightsError::TrailingBytes {
                    expected: counts.wcls * std::mem::size_of::<f32>(),
                    actual,
                });
            }
        };

        Ok(Self {
            token_embedding_table,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            freq_cis_real,
            freq_cis_imag,
            wcls,
        })
    }

    pub fn from_model_file(
        path: impl AsRef<Path>,
        config: &Config,
    ) -> Result<Self, WeightsError> {
        let file = File::open(path).map_err(WeightsError::Io)?;
        let mut reader = BufReader::new(file);
        reader
            .seek(SeekFrom::Start(Config::header_size() as u64))
            .map_err(WeightsError::Io)?;
        Self::from_reader(config, &mut reader)
    }

    pub fn classifier_weights(&self) -> &[f32] {
        self.wcls
            .as_deref()
            .unwrap_or(self.token_embedding_table.as_slice())
    }
}

impl WeightCounts {
    pub fn from_config(config: &Config) -> Result<Self, WeightsError> {
        let head_dim = config.head_dim();
        if head_dim % 2 != 0 {
            return Err(WeightsError::InvalidLayout(
                "head_dim must be even to represent RoPE frequencies",
            ));
        }

        let rope_span = checked_mul(config.seq_len, head_dim / 2, "freq_cis")?;
        let layer_dim = checked_mul(config.n_layers, config.dim, "layer_dim")?;
        let layer_q = checked_mul(config.n_layers, checked_mul(config.dim, config.dim, "wq")?, "wq")?;
        let kv_proj = checked_mul(config.dim, config.kv_dim(), "kv_proj")?;
        let layer_kv = checked_mul(config.n_layers, kv_proj, "kv_proj")?;
        let ffn_proj = checked_mul(config.hidden_dim, config.dim, "ffn_proj")?;
        let layer_ffn = checked_mul(config.n_layers, ffn_proj, "ffn_proj")?;
        let embedding = checked_mul(config.vocab_size, config.dim, "embedding")?;

        Ok(Self {
            token_embedding_table: embedding,
            rms_att_weight: layer_dim,
            wq: layer_q,
            wk: layer_kv,
            wv: layer_kv,
            wo: layer_q,
            rms_ffn_weight: layer_dim,
            w1: layer_ffn,
            w2: layer_ffn,
            w3: layer_ffn,
            rms_final_weight: config.dim,
            freq_cis_real: rope_span,
            freq_cis_imag: rope_span,
            wcls: embedding,
        })
    }

    pub fn required_f32_count(&self) -> usize {
        self.token_embedding_table
            + self.rms_att_weight
            + self.wq
            + self.wk
            + self.wv
            + self.wo
            + self.rms_ffn_weight
            + self.w1
            + self.w2
            + self.w3
            + self.rms_final_weight
            + self.freq_cis_real
            + self.freq_cis_imag
    }
}

impl fmt::Display for WeightsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to read weights: {err}"),
            Self::InvalidLayout(reason) => write!(f, "invalid weight layout: {reason}"),
            Self::TrailingBytes { expected, actual } => write!(
                f,
                "unexpected trailing bytes after mandatory weights: expected {expected}, got {actual}"
            ),
            Self::SizeOverflow(name) => write!(f, "weight size overflow while computing `{name}`"),
        }
    }
}

impl std::error::Error for WeightsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidLayout(_) | Self::TrailingBytes { .. } | Self::SizeOverflow(_) => None,
        }
    }
}

fn read_f32_tensor(reader: &mut impl Read, count: usize) -> Result<Vec<f32>, WeightsError> {
    let byte_len = checked_mul(count, std::mem::size_of::<f32>(), "tensor_bytes")?;
    let mut bytes = vec![0_u8; byte_len];
    reader.read_exact(&mut bytes).map_err(WeightsError::Io)?;
    bytes_to_f32_vec(bytes)
}

fn bytes_to_f32_vec(bytes: Vec<u8>) -> Result<Vec<f32>, WeightsError> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(WeightsError::InvalidLayout(
            "tensor byte length must be a multiple of 4",
        ));
    }

    Ok(bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk size is fixed")))
        .collect())
}

fn checked_mul(lhs: usize, rhs: usize, name: &'static str) -> Result<usize, WeightsError> {
    lhs.checked_mul(rhs).ok_or(WeightsError::SizeOverflow(name))
}

#[cfg(test)]
mod tests {
    use super::{WeightCounts, Weights, WeightsError};
    use crate::config::Config;

    fn test_config() -> Config {
        Config {
            dim: 8,
            hidden_dim: 16,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            vocab_size: 10,
            seq_len: 4,
        }
    }

    fn floats_to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    }

    #[test]
    fn loads_required_and_optional_tensors() {
        let config = test_config();
        let counts = WeightCounts::from_config(&config).expect("counts should compute");
        let total = counts.required_f32_count() + counts.wcls;
        let raw = (0..total).map(|value| value as f32).collect::<Vec<_>>();
        let bytes = floats_to_bytes(&raw);
        let mut reader = bytes.as_slice();

        let weights = Weights::from_reader(&config, &mut reader).expect("weights should load");

        assert_eq!(weights.token_embedding_table.len(), counts.token_embedding_table);
        assert_eq!(weights.freq_cis_real.len(), counts.freq_cis_real);
        assert_eq!(weights.wcls.as_ref().map(Vec::len), Some(counts.wcls));
        assert_eq!(weights.token_embedding_table[0], 0.0);
        assert_eq!(
            weights.wcls.as_ref().expect("wcls should exist")[counts.wcls - 1],
            (total - 1) as f32
        );
    }

    #[test]
    fn allows_missing_wcls_and_falls_back_to_embeddings() {
        let config = test_config();
        let counts = WeightCounts::from_config(&config).expect("counts should compute");
        let raw = (0..counts.required_f32_count())
            .map(|value| value as f32)
            .collect::<Vec<_>>();
// @trace-pilot a8727e84ecdf804978f3e9cd2100ed5d43740b4f
        let bytes = floats_to_bytes(&raw);
        let mut reader = bytes.as_slice();

        let weights = Weights::from_reader(&config, &mut reader).expect("weights should load");

        assert!(weights.wcls.is_none());
        assert_eq!(
            weights.classifier_weights(),
            weights.token_embedding_table.as_slice()
        );
    }

    #[test]
    fn rejects_invalid_trailing_bytes() {
        let config = test_config();
        let counts = WeightCounts::from_config(&config).expect("counts should compute");
        let mut bytes = floats_to_bytes(
            &(0..counts.required_f32_count())
                .map(|value| value as f32)
                .collect::<Vec<_>>(),
        );
        bytes.extend_from_slice(&[1, 2, 3, 4, 5]);
        let mut reader = bytes.as_slice();

        let err = Weights::from_reader(&config, &mut reader).expect_err("weights should fail");
        assert!(matches!(err, WeightsError::TrailingBytes { .. }));
    }

    #[test]
    fn rejects_odd_head_dim_for_rope_tables() {
        let config = Config {
            dim: 6,
            hidden_dim: 16,
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            vocab_size: 10,
            seq_len: 4,
        };

        let err = WeightCounts::from_config(&config).expect_err("counts should fail");
        assert!(matches!(err, WeightsError::InvalidLayout(_)));
    }
}
