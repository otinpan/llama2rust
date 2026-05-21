use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

use crate::{
    config::Config,
    weights::{LayerWeights, TransformerWeights},
};

#[derive(Debug)]
pub struct Checkpoint {
    pub config: Config,
    pub weights: TransformerWeights,
    pub shared_weights: bool,
}

pub fn load_checkpoint(path: &Path) -> io::Result<Checkpoint> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    load_checkpoint_from_reader(&mut reader)
}

fn load_checkpoint_from_reader<R: Read>(reader: &mut R) -> io::Result<Checkpoint> {
    let mut header = [0_u8; Config::BYTE_SIZE];
    reader.read_exact(&mut header)?;
    let (config, shared_weights) = Config::from_bytes(header)?;

    let mut weights = TransformerWeights::new(&config);
    read_f32_slice(reader, &mut weights.token_embedding_table)?;

    for layer in &mut weights.layers {
        read_layer_weights(reader, layer)?;
    }

    read_f32_slice(reader, &mut weights.rms_final_weight)?;

    let rope_table_size = config.seq_len * config.head_size() / 2;
    skip_f32s(reader, rope_table_size)?;
    skip_f32s(reader, rope_table_size)?;

    if !shared_weights {
        read_f32_slice(reader, &mut weights.wcls)?;
    } else {
        weights.wcls.copy_from_slice(&weights.token_embedding_table);
    }

    Ok(Checkpoint {
        config,
        weights,
        shared_weights,
    })
}

fn read_layer_weights<R: Read>(reader: &mut R, layer: &mut LayerWeights) -> io::Result<()> {
    read_f32_slice(reader, &mut layer.rms_att_weight)?;
    read_f32_slice(reader, &mut layer.wq)?;
    read_f32_slice(reader, &mut layer.wk)?;
    read_f32_slice(reader, &mut layer.wv)?;
    read_f32_slice(reader, &mut layer.wo)?;
    read_f32_slice(reader, &mut layer.rms_ffn_weight)?;
    read_f32_slice(reader, &mut layer.w1)?;
    read_f32_slice(reader, &mut layer.w2)?;
    read_f32_slice(reader, &mut layer.w3)?;
    Ok(())
}

fn skip_f32s<R: Read>(reader: &mut R, count: usize) -> io::Result<()> {
    let mut bytes = vec![0_u8; count * std::mem::size_of::<f32>()];
    reader.read_exact(&mut bytes)?;
    Ok(())
}

fn read_f32_slice<R: Read>(reader: &mut R, out: &mut [f32]) -> io::Result<()> {
    let mut bytes = vec![0_u8; std::mem::size_of_val(out)];
    reader.read_exact(&mut bytes)?;

    for (slot, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
        *slot = f32::from_le_bytes(chunk.try_into().expect("4-byte chunk"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use crate::config::Config;

    use super::load_checkpoint_from_reader;

    #[test]
    fn loads_unshared_classifier_weights() {
        let config = Config::new(4, 8, 1, 2, 2, 3, 4);
        let raw = fixture_bytes(&config, false);
        let mut cursor = Cursor::new(raw);

        let checkpoint = load_checkpoint_from_reader(&mut cursor).unwrap();
        let rms_final_start = (config.vocab_size * config.dim
            + config.n_layers * config.dim
            + config.n_layers * config.dim * config.dim
            + config.n_layers * config.dim * config.kv_dim()
            + config.n_layers * config.dim * config.kv_dim()
            + config.n_layers * config.dim * config.dim
            + config.n_layers * config.dim
            + config.n_layers * config.hidden_dim * config.dim
            + config.n_layers * config.dim * config.hidden_dim
            + config.n_layers * config.hidden_dim * config.dim
            + 1) as f32;
        let wcls_start = (rms_final_start as usize
            + config.dim
            + (config.seq_len * config.head_size() / 2) * 2) as f32;

        assert!(!checkpoint.shared_weights);
        assert_eq!(checkpoint.weights.token_embedding_table[0], 1.0);
        assert_eq!(checkpoint.weights.layers[0].rms_att_weight[0], 13.0);
        assert_eq!(checkpoint.weights.rms_final_weight[0], rms_final_start);
        assert_eq!(checkpoint.weights.wcls[0], wcls_start);
    }

    #[test]
    fn reuses_embedding_table_when_classifier_is_shared() {
        let config = Config::new(4, 8, 1, 2, 2, 3, 4);
        let raw = fixture_bytes(&config, true);
        let mut cursor = Cursor::new(raw);

        let checkpoint = load_checkpoint_from_reader(&mut cursor).unwrap();

        assert!(checkpoint.shared_weights);
        assert_eq!(
            checkpoint.weights.wcls,
            checkpoint.weights.token_embedding_table
        );
    }

    fn fixture_bytes(config: &Config, shared_weights: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        let vocab_marker = if shared_weights {
            config.vocab_size as i32
        } else {
            -(config.vocab_size as i32)
        };
        let header = [
            config.dim as i32,
            config.hidden_dim as i32,
            config.n_layers as i32,
            config.n_heads as i32,
            config.n_kv_heads as i32,
            vocab_marker,
            config.seq_len as i32,
        ];

        for value in header {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let kv_dim = config.kv_dim();
        let rope_size = config.seq_len * config.head_size() / 2;
        let mut next = 1.0_f32;

        push_f32s(&mut bytes, &mut next, config.vocab_size * config.dim);
        push_f32s(&mut bytes, &mut next, config.n_layers * config.dim);
        push_f32s(&mut bytes, &mut next, config.n_layers * config.dim * config.dim);
        push_f32s(&mut bytes, &mut next, config.n_layers * config.dim * kv_dim);
        push_f32s(&mut bytes, &mut next, config.n_layers * config.dim * kv_dim);
        push_f32s(&mut bytes, &mut next, config.n_layers * config.dim * config.dim);
        push_f32s(&mut bytes, &mut next, config.n_layers * config.dim);
        push_f32s(
            &mut bytes,
            &mut next,
            config.n_layers * config.hidden_dim * config.dim,
        );
        push_f32s(
            &mut bytes,
            &mut next,
            config.n_layers * config.dim * config.hidden_dim,
        );
        push_f32s(
            &mut bytes,
            &mut next,
            config.n_layers * config.hidden_dim * config.dim,
        );
        push_f32s(&mut bytes, &mut next, config.dim);
        push_f32s(&mut bytes, &mut next, rope_size);
        push_f32s(&mut bytes, &mut next, rope_size);

        if !shared_weights {
            push_f32s(&mut bytes, &mut next, config.vocab_size * config.dim);
        }

        bytes
    }

    fn push_f32s(bytes: &mut Vec<u8>, next: &mut f32, count: usize) {
        for _ in 0..count {
            bytes.extend_from_slice(&next.to_le_bytes());
            *next += 1.0;
        }
    }
}
