use crate::{
    config::Config,
    kernels::{accum, matmul, rmsnorm, softmax, swiglu},
    state::RunState,
    weights::TransformerWeights,
};

#[derive(Debug)]
pub struct Transformer {
    config: Config,
    weights: TransformerWeights,
    state: RunState,
}

impl Transformer {
    pub fn new(config: Config, weights: TransformerWeights) -> Self {
        let state = RunState::new(&config);
        Self {
            config,
            weights,
            state,
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn weights(&self) -> &TransformerWeights {
        &self.weights
    }

    pub fn state(&self) -> &RunState {
        &self.state
    }

    pub fn reset_state(&mut self) {
        self.state = RunState::new(&self.config);
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        assert!(token < self.config.vocab_size, "token out of range");
        assert!(pos < self.config.seq_len, "position out of range");

        let dim = self.config.dim;
        let token_start = token * dim;
        let token_end = token_start + dim;
        self.state
            .x
            .copy_from_slice(&self.weights.token_embedding_table[token_start..token_end]);

        for layer in 0..self.config.n_layers {
            self.forward_layer_inner(pos, layer);
        }

        let x = self.state.x.clone();
        rmsnorm(&mut self.state.x, &x, &self.weights.rms_final_weight);
        matmul(
            &mut self.state.logits,
            &self.weights.wcls,
            &self.state.x,
            self.config.vocab_size,
            dim,
        );

        &self.state.logits
    }

    pub fn forward_layer(&mut self, token: usize, pos: usize, layer: usize) -> &[f32] {
        assert!(token < self.config.vocab_size, "token out of range");
        assert!(pos < self.config.seq_len, "position out of range");
        assert!(layer < self.config.n_layers, "layer out of range");

        let dim = self.config.dim;
        let token_start = token * dim;
        let token_end = token_start + dim;
        self.state
            .x
            .copy_from_slice(&self.weights.token_embedding_table[token_start..token_end]);

        self.forward_layer_inner(pos, layer);

        &self.state.x
    }

    fn forward_layer_inner(&mut self, pos: usize, layer: usize) {
        let dim = self.config.dim;
        let kv_dim = self.config.kv_dim();
        let kv_mul = self.config.n_heads / self.config.n_kv_heads;
        let hidden_dim = self.config.hidden_dim;
        let head_size = self.config.head_size();
        let weights = &self.weights.layers[layer];

        rmsnorm(&mut self.state.xb, &self.state.x, &weights.rms_att_weight);

        matmul(&mut self.state.q, &weights.wq, &self.state.xb, dim, dim);
        matmul(&mut self.state.k, &weights.wk, &self.state.xb, kv_dim, dim);
        matmul(&mut self.state.v, &weights.wv, &self.state.xb, kv_dim, dim);

        apply_rope_q_and_k(&mut self.state.q, &mut self.state.k, pos, head_size, kv_dim);

        let loff = layer * self.config.seq_len * kv_dim;
        let cache_offset = loff + pos * kv_dim;
        self.state.key_cache[cache_offset..cache_offset + kv_dim].copy_from_slice(&self.state.k);
        self.state.value_cache[cache_offset..cache_offset + kv_dim].copy_from_slice(&self.state.v);

        for head in 0..self.config.n_heads {
            let q = &self.state.q[head * head_size..(head + 1) * head_size];
            let att =
                &mut self.state.att[head * self.config.seq_len..(head + 1) * self.config.seq_len];

            for t in 0..=pos {
                let key_offset = loff + t * kv_dim + (head / kv_mul) * head_size;
                let k = &self.state.key_cache[key_offset..key_offset + head_size];
                let score = q.iter().zip(k.iter()).map(|(qv, kv)| qv * kv).sum::<f32>()
                    / (head_size as f32).sqrt();
                att[t] = score;
            }

            softmax(&mut att[..=pos]);

            let xb = &mut self.state.xb[head * head_size..(head + 1) * head_size];
            xb.fill(0.0);
            for t in 0..=pos {
                let value_offset = loff + t * kv_dim + (head / kv_mul) * head_size;
                let v = &self.state.value_cache[value_offset..value_offset + head_size];
                let a = att[t];
                for (dst, value) in xb.iter_mut().zip(v.iter()) {
                    *dst += a * value;
                }
            }
        }

        matmul(&mut self.state.xb2, &weights.wo, &self.state.xb, dim, dim);
        accum(&mut self.state.x, &self.state.xb2);

        rmsnorm(&mut self.state.xb, &self.state.x, &weights.rms_ffn_weight);
        matmul(&mut self.state.hb, &weights.w1, &self.state.xb, hidden_dim, dim);
        matmul(&mut self.state.hb2, &weights.w3, &self.state.xb, hidden_dim, dim);
        let gate = self.state.hb.clone();
        swiglu(&mut self.state.hb, &gate, &self.state.hb2);
        matmul(&mut self.state.xb, &weights.w2, &self.state.hb, dim, hidden_dim);
        accum(&mut self.state.x, &self.state.xb);
    }
}

fn apply_rope_q_and_k(q: &mut [f32], k: &mut [f32], pos: usize, head_size: usize, kv_dim: usize) {
    for i in (0..q.len()).step_by(2) {
        let head_dim = i % head_size;
        let freq = 1.0_f32 / 10000.0_f32.powf(head_dim as f32 / head_size as f32);
        let angle = pos as f32 * freq;
        let cos = angle.cos();
        let sin = angle.sin();

        rotate_pair(q, i, cos, sin);
        if i < kv_dim {
            rotate_pair(k, i, cos, sin);
        }
    }
}

fn rotate_pair(values: &mut [f32], index: usize, cos: f32, sin: f32) {
    let v0 = values[index];
    let v1 = values[index + 1];
    values[index] = v0 * cos - v1 * sin;
    values[index + 1] = v0 * sin + v1 * cos;
}

#[cfg(test)]
mod tests {
    use crate::{config::Config, weights::TransformerWeights};

    use super::Transformer;

    #[test]
    fn owns_weights_and_runtime_state() {
        let config = Config::new(64, 128, 2, 4, 2, 256, 32);
        let weights = TransformerWeights::new(&config);
        let transformer = Transformer::new(config, weights);

        assert_eq!(transformer.config().head_size(), 16);
        assert_eq!(transformer.weights().layers.len(), 2);
        assert_eq!(transformer.state().x.len(), 64);
    }

    #[test]
    fn runs_single_layer_and_updates_cache() {
        let config = Config::new(4, 6, 1, 2, 2, 8, 4);
        let mut weights = TransformerWeights::new(&config);
        weights.token_embedding_table[4..8].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);

        let layer = &mut weights.layers[0];
        layer.rms_att_weight.fill(1.0);
        layer.rms_ffn_weight.fill(1.0);
        layer.wq.copy_from_slice(&identity_matrix(4));
        layer.wk.copy_from_slice(&identity_matrix(4));
        layer.wv.copy_from_slice(&identity_matrix(4));
        layer.wo.copy_from_slice(&identity_matrix(4));
        layer.w1.copy_from_slice(&rectangular_identity(6, 4));
        layer.w2.copy_from_slice(&rectangular_identity(4, 6));
        layer.w3.copy_from_slice(&rectangular_identity(6, 4));

        let mut transformer = Transformer::new(config, weights);
        let output = transformer.forward_layer(1, 0, 0).to_vec();

        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| value.abs() > 0.0));
        assert_eq!(&transformer.state().key_cache[..4], &transformer.state().k);
        assert_eq!(&transformer.state().value_cache[..4], &transformer.state().v);
    }

    #[test]
    fn runs_full_forward_and_returns_logits() {
        let config = Config::new(4, 6, 2, 2, 2, 8, 4);
        let mut weights = TransformerWeights::new(&config);
        weights.token_embedding_table[4..8].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        weights.rms_final_weight.fill(1.0);
        weights.wcls.copy_from_slice(&rectangular_identity(8, 4));

        for layer in &mut weights.layers {
            layer.rms_att_weight.fill(1.0);
            layer.rms_ffn_weight.fill(1.0);
            layer.wq.copy_from_slice(&identity_matrix(4));
            layer.wk.copy_from_slice(&identity_matrix(4));
            layer.wv.copy_from_slice(&identity_matrix(4));
            layer.wo.copy_from_slice(&identity_matrix(4));
            layer.w1.copy_from_slice(&rectangular_identity(6, 4));
            layer.w2.copy_from_slice(&rectangular_identity(4, 6));
            layer.w3.copy_from_slice(&rectangular_identity(6, 4));
        }

        let mut transformer = Transformer::new(config, weights);
        let logits = transformer.forward(1, 0).to_vec();

        assert_eq!(logits.len(), 8);
        assert!(logits.iter().all(|value| value.is_finite()));
        assert!(logits.iter().any(|value| value.abs() > 0.0));
    }

    fn identity_matrix(size: usize) -> Vec<f32> {
        let mut values = vec![0.0; size * size];
        for index in 0..size {
            values[index * size + index] = 1.0;
        }
        values
    }

    fn rectangular_identity(rows: usize, cols: usize) -> Vec<f32> {
        let mut values = vec![0.0; rows * cols];
        for index in 0..rows.min(cols) {
            values[index * cols + index] = 1.0;
        }
        values
    }
}
