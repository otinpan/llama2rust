use crate::config::Config;

#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub rms_att_weight: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub rms_ffn_weight: Vec<f32>,
    pub w1: Vec<f32>,
    pub w2: Vec<f32>,
    pub w3: Vec<f32>,
}

impl LayerWeights {
    pub fn new(config: &Config) -> Self {
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let kv_dim = config.kv_dim();

        Self {
            rms_att_weight: vec![0.0; dim],
            wq: vec![0.0; dim * dim],
            wk: vec![0.0; dim * kv_dim],
            wv: vec![0.0; dim * kv_dim],
            wo: vec![0.0; dim * dim],
            rms_ffn_weight: vec![0.0; dim],
            w1: vec![0.0; hidden_dim * dim],
            w2: vec![0.0; dim * hidden_dim],
            w3: vec![0.0; hidden_dim * dim],
        }
    }
}

#[derive(Debug, Clone)]
pub struct TransformerWeights {
    pub token_embedding_table: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub rms_final_weight: Vec<f32>,
    pub wcls: Vec<f32>,
}

impl TransformerWeights {
    pub fn new(config: &Config) -> Self {
        let layers = (0..config.n_layers)
            .map(|_| LayerWeights::new(config))
            .collect();

        Self {
            token_embedding_table: vec![0.0; config.vocab_size * config.dim],
            layers,
            rms_final_weight: vec![0.0; config.dim],
            wcls: vec![0.0; config.vocab_size * config.dim],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::config::Config;

    use super::TransformerWeights;

    #[test]
    fn allocates_expected_shapes() {
        let config = Config::new(64, 128, 2, 4, 2, 256, 32);
        let weights = TransformerWeights::new(&config);

        assert_eq!(weights.token_embedding_table.len(), 16_384);
        assert_eq!(weights.layers.len(), 2);
        assert_eq!(weights.layers[0].wk.len(), 2_048);
        assert_eq!(weights.wcls.len(), 16_384);
    }
}
