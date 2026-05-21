use crate::config::Config;

#[derive(Debug, Clone)]
pub struct RunState {
    pub x: Vec<f32>,
    pub xb: Vec<f32>,
    pub xb2: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub att: Vec<f32>,
    pub logits: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
}

impl RunState {
    pub fn new(config: &Config) -> Self {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let kv_dim = config.kv_dim();

        Self {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            xb2: vec![0.0; dim],
            hb: vec![0.0; hidden_dim],
            hb2: vec![0.0; hidden_dim],
            q: vec![0.0; dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: vec![0.0; config.n_heads * config.seq_len],
            logits: vec![0.0; config.vocab_size],
            key_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
            value_cache: vec![0.0; config.n_layers * config.seq_len * kv_dim],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::Config;

    use super::RunState;

    #[test]
    fn allocates_expected_runtime_buffers() {
        let config = Config::new(64, 128, 2, 4, 2, 256, 32);
        let state = RunState::new(&config);

        assert_eq!(state.att.len(), 128);
        assert_eq!(state.logits.len(), 256);
        assert_eq!(state.key_cache.len(), 2_048);
        assert_eq!(state.value_cache.len(), 2_048);
    }
}
