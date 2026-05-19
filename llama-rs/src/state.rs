// state
// @trace-pilot 121565c1eb74536f6c9f4ec8c61ccc49665dc9c3

use crate::config::Config;

#[derive(Debug)]
pub struct RunState{
    // current hidden state
    pub x: Vec<f32>,

    // residual branch buffers
    pub xb: Vec<f32>,
    pub xb2: Vec<f32>,

    // FFN hidden buffers
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,

    // attention
    // @trace-pilot 04abe07bcf4f298e4692bcce03d84966fd63802c
    pub q: Vec<f32>,

    // attention score
    pub att: Vec<f32>,

    // final logits
    pub logits: Vec<f32>,

    // KV cache
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
}

impl RunState {
    // @trace-pilot e28b627b4913c920625528b09d582759910a5f33
    // void malloc_run_state
    pub fn new(config: &Config) -> Self {
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        Self {
            x: vec![0.0; config.dim],
            xb: vec![0.0; config.dim],
            xb2: vec![0.0; config.dim],
            hb: vec![0.0; config.hidden_dim],
            hb2: vec![0.0; config.hidden_dim],
            q: vec![0.0; config.dim],
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
            value_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
        }
    }
}
