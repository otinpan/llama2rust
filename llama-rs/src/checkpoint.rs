use std::fs::File;
use std::io;
use std::path::Path;

use memmap2::Mmap;

use crate::config::Config;
use crate::weights::TransformerWeights;

const CONFIG_FIELDS: usize = 7;
const CONFIG_BYTES: usize = CONFIG_FIELDS * std::mem::size_of::<i32>();

#[derive(Debug)]
pub struct Checkpoint {
    mmap: Mmap,
    config: Config,
    shared_weights: bool,
}

impl Checkpoint {
    // @trace-pilot 1ccebadb31d03a8b955448590f676976da8baebb
    // void read_checkpoint
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let (config, shared_weights) = read_config(&mmap)?;
        Ok(Self {
            mmap,
            config,
            shared_weights,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
    // @trace-pilot 60447ae71497cf3c9dfcd0332e32c42504bc4b1f
    // void memory_map_weights
    pub fn weights(&self) -> io::Result<TransformerWeights<'_>> {
        memory_map_weights(&self.mmap, &self.config, self.shared_weights)
    }
}

fn read_config(bytes: &[u8]) -> io::Result<(Config, bool)> {
    if bytes.len() < CONFIG_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "checkpoint is smaller than config header",
        ));
    }

    let mut fields = [0_i32; CONFIG_FIELDS];
    for (index, field) in fields.iter_mut().enumerate() {
        let start = index * 4;
        let end = start + 4;
        *field = i32::from_le_bytes(bytes[start..end].try_into().unwrap());
    }

    let shared_weights = fields[5] > 0;
    let vocab_size = fields[5].unsigned_abs() as usize;

    Ok((
        Config {
            dim: fields[0] as usize,
            hidden_dim: fields[1] as usize,
            n_layers: fields[2] as usize,
            n_heads: fields[3] as usize,
            n_kv_heads: fields[4] as usize,
            vocab_size,
            seq_len: fields[6] as usize,
        },
        shared_weights,
    ))
}

// Transformerはmmapと同じ寿命を持つ
// weightsが必要なときに呼ぶ
fn memory_map_weights<'a>(
    mmap: &'a Mmap,
    config: &Config,
    shared_weights: bool,
) -> io::Result<TransformerWeights<'a>> {
    let data = bytes_as_f32_slice(&mmap[CONFIG_BYTES..])?;
    let head_size = config.dim / config.n_heads;
    let n_layers = config.n_layers;
    let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    let rope_cache = config.seq_len * head_size / 2;
    let mut cursor = WeightCursor::new(data);

    let token_embedding_table = cursor.take(config.vocab_size * config.dim)?;
    let rms_att_weight = cursor.take(n_layers * config.dim)?;
    let wq = cursor.take(n_layers * config.dim * config.dim)?;
    let wk = cursor.take(n_layers * config.dim * kv_dim)?;
    let wv = cursor.take(n_layers * config.dim * kv_dim)?;
    let wo = cursor.take(n_layers * config.dim * config.dim)?;
    let rms_ffn_weight = cursor.take(n_layers * config.dim)?;
    let w1 = cursor.take(n_layers * config.dim * config.hidden_dim)?;
    let w2 = cursor.take(n_layers * config.hidden_dim * config.dim)?;
    let w3 = cursor.take(n_layers * config.dim * config.hidden_dim)?;
    let rms_final_weight = cursor.take(config.dim)?;
    cursor.take(rope_cache)?;
    cursor.take(rope_cache)?;
    let wcls = if shared_weights {
        token_embedding_table
    } else {
        cursor.take(config.vocab_size * config.dim)?
    };

    Ok(TransformerWeights {
        token_embedding_table,
        rms_att_weight,
        rms_ffn_weight,
        wq,
        wk,
        wv,
        wo,
        w1,
        w2,
        w3,
        rms_final_weight,
        wcls,
    })
}

fn bytes_as_f32_slice(bytes: &[u8]) -> io::Result<&[f32]> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint weight section is not aligned to f32 values",
        ));
    }

    let (_, floats, remainder) = unsafe { bytes.align_to::<f32>() };
    if !remainder.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "checkpoint weight section has trailing non-f32 bytes",
        ));
    }
    Ok(floats)
}

struct WeightCursor<'a> {
    data: &'a [f32],
    offset: usize,
}

impl<'a> WeightCursor<'a> {
    fn new(data: &'a [f32]) -> Self {
        Self { data, offset: 0 }
    }

    fn take(&mut self, len: usize) -> io::Result<&'a [f32]> {
        let end = self.offset.checked_add(len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "checkpoint weight size overflow")
        })?;
        if end > self.data.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "checkpoint ended before all weights were read",
            ));
        }
        let slice = &self.data[self.offset..end];
        self.offset = end;
        Ok(slice)
    }
}
