// @trace-pilot 173259109bcbd478d15d87542b3f76643fb65c0f
use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

const CONFIG_FIELD_COUNT: usize = 7;

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

#[derive(Debug)]
pub enum ConfigError {
    Io(io::Error),
    InvalidField {
        name: &'static str,
        value: i32,
        reason: &'static str,
    },
}

impl Config {
    pub fn from_reader(reader: &mut impl Read) -> Result<Self, ConfigError> {
        let dim = read_positive_i32(reader, "dim")?;
        let hidden_dim = read_positive_i32(reader, "hidden_dim")?;
        let n_layers = read_positive_i32(reader, "n_layers")?;
        let n_heads = read_positive_i32(reader, "n_heads")?;
        let n_kv_heads = read_positive_i32(reader, "n_kv_heads")?;
        let vocab_size = read_positive_i32(reader, "vocab_size")?;
        let seq_len = read_positive_i32(reader, "seq_len")?;

        let config = Self {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        };

        config.validate()?;
        Ok(config)
    }

    pub fn from_model_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let file = File::open(path).map_err(ConfigError::Io)?;
        let mut reader = BufReader::new(file);
        Self::from_reader(&mut reader)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.n_heads == 0 {
            return Err(ConfigError::InvalidField {
                name: "n_heads",
                value: 0,
                reason: "must be greater than zero",
            });
        }

        if self.n_kv_heads == 0 {
            return Err(ConfigError::InvalidField {
                name: "n_kv_heads",
                value: 0,
                reason: "must be greater than zero",
            });
        }

        if self.dim % self.n_heads != 0 {
            return Err(ConfigError::InvalidField {
                name: "dim",
                value: self.dim as i32,
                reason: "must be divisible by n_heads",
            });
        }

        if self.n_heads % self.n_kv_heads != 0 {
            return Err(ConfigError::InvalidField {
                name: "n_heads",
                value: self.n_heads as i32,
                reason: "must be divisible by n_kv_heads",
            });
        }

        Ok(())
    }

    pub const fn header_size() -> usize {
        CONFIG_FIELD_COUNT * std::mem::size_of::<i32>()
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.head_dim() * self.n_kv_heads
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to read model config: {err}"),
            Self::InvalidField {
                name,
                value,
                reason,
            } => write!(f, "invalid config field `{name}` ({value}): {reason}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidField { .. } => None,
        }
    }
}

fn read_positive_i32(
    reader: &mut impl Read,
    name: &'static str,
) -> Result<usize, ConfigError> {
    let value = read_i32(reader).map_err(ConfigError::Io)?;
    if value <= 0 {
        return Err(ConfigError::InvalidField {
            name,
            value,
            reason: "must be greater than zero",
        });
    }

    Ok(value as usize)
}

fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::{Config, ConfigError};

    fn config_bytes(fields: [i32; 7]) -> Vec<u8> {
        fields
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<u8>>()
    }

    #[test]
    fn reads_header_into_config() {
        let raw = config_bytes([4096, 11008, 32, 32, 32, 32000, 2048]);
        let mut bytes = raw.as_slice();
        let config = Config::from_reader(&mut bytes).expect("config should parse");

        assert_eq!(config.dim, 4096);
        assert_eq!(config.hidden_dim, 11008);
        assert_eq!(config.n_layers, 32);
        assert_eq!(config.n_heads, 32);
        assert_eq!(config.n_kv_heads, 32);
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.seq_len, 2048);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.kv_dim(), 4096);
    }

    #[test]
    fn rejects_non_positive_fields() {
// @trace-pilot 173259109bcbd478d15d87542b3f76643fb65c0f
        let raw = config_bytes([4096, 11008, 32, 0, 32, 32000, 2048]);
        let mut bytes = raw.as_slice();
        let err = Config::from_reader(&mut bytes).expect_err("config should fail");

        assert!(matches!(
            err,
            ConfigError::InvalidField {
                name: "n_heads",
                value: 0,
                ..
            }
        ));
    }

    #[test]
    fn rejects_invalid_head_relationships() {
// @trace-pilot 173259109bcbd478d15d87542b3f76643fb65c0f
        let raw = config_bytes([4097, 11008, 32, 32, 8, 32000, 2048]);
        let mut bytes = raw.as_slice();
        let err = Config::from_reader(&mut bytes).expect_err("config should fail");

        assert!(matches!(
            err,
            ConfigError::InvalidField {
                name: "dim",
                reason: "must be divisible by n_heads",
                ..
            }
        ));
    }
}
