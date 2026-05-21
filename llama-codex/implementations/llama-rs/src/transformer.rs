// @trace-pilot 4ad8b2c1ac45cf708430b9574ecae2a1dc28081b
use std::fmt;

use crate::config::Config;
use crate::weights::{WeightCounts, Weights, WeightsError};

const RMS_NORM_EPS: f32 = 1.0e-5;

#[derive(Debug, Clone)]
pub struct Transformer {
    config: Config,
    weights: Weights,
    state: RunState,
    kv_mul: usize,
}

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

#[derive(Debug)]
pub enum TransformerError {
    InvalidToken {
        token: u32,
        vocab_size: usize,
    },
    InvalidPosition {
        position: usize,
        seq_len: usize,
    },
    InvalidWeights(&'static str),
    WeightLayout(WeightsError),
}

impl Transformer {
    pub fn new(config: Config, weights: Weights) -> Result<Self, TransformerError> {
        validate_weight_shapes(&config, &weights)?;
        let kv_mul = config.n_heads / config.n_kv_heads;
        let state = RunState::new(&config);
        Ok(Self {
            config,
            weights,
            state,
            kv_mul,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn state(&self) -> &RunState {
        &self.state
    }

    pub fn forward(&mut self, token: u32, position: usize) -> Result<&[f32], TransformerError> {
        if token as usize >= self.config.vocab_size {
            return Err(TransformerError::InvalidToken {
                token,
                vocab_size: self.config.vocab_size,
            });
        }

        if position >= self.config.seq_len {
            return Err(TransformerError::InvalidPosition {
                position,
                seq_len: self.config.seq_len,
            });
        }

        let config = &self.config;
        let weights = &self.weights;
        let state = &mut self.state;
        let dim = config.dim;
        let hidden_dim = config.hidden_dim;
        let head_dim = config.head_dim();
        let kv_dim = config.kv_dim();
        let token_offset = token as usize * dim;

        state.x.copy_from_slice(&weights.token_embedding_table[token_offset..token_offset + dim]);

        for layer in 0..config.n_layers {
            let layer_dim = layer * dim;
            let wq_offset = layer * dim * dim;
            let wk_offset = layer * kv_dim * dim;
            let ffn_offset = layer * hidden_dim * dim;
            let w2_offset = layer * dim * hidden_dim;
            let freq_offset = position * (head_dim / 2);

            rmsnorm(
                &mut state.xb,
                &state.x,
                &weights.rms_att_weight[layer_dim..layer_dim + dim],
            );

            matmul(
                &weights.wq[wq_offset..wq_offset + dim * dim],
                &state.xb,
                &mut state.q,
                dim,
                dim,
            );
            matmul(
                &weights.wk[wk_offset..wk_offset + kv_dim * dim],
                &state.xb,
                &mut state.k,
                kv_dim,
                dim,
            );
            matmul(
                &weights.wv[wk_offset..wk_offset + kv_dim * dim],
                &state.xb,
                &mut state.v,
                kv_dim,
                dim,
            );

            apply_rope(
                &mut state.q,
                &weights.freq_cis_real[freq_offset..freq_offset + head_dim / 2],
                &weights.freq_cis_imag[freq_offset..freq_offset + head_dim / 2],
                config.n_heads,
                head_dim,
            );
            apply_rope(
                &mut state.k,
                &weights.freq_cis_real[freq_offset..freq_offset + head_dim / 2],
                &weights.freq_cis_imag[freq_offset..freq_offset + head_dim / 2],
                config.n_kv_heads,
                head_dim,
            );

            cache_kv(state, config, layer, position);
            attention(state, config, self.kv_mul, layer, position);

            matmul(
                &weights.wo[wq_offset..wq_offset + dim * dim],
                &state.xb,
                &mut state.xb2,
                dim,
                dim,
            );
            for i in 0..dim {
                state.x[i] += state.xb2[i];
            }

            rmsnorm(
                &mut state.xb,
                &state.x,
                &weights.rms_ffn_weight[layer_dim..layer_dim + dim],
            );

            matmul(
                &weights.w1[ffn_offset..ffn_offset + hidden_dim * dim],
                &state.xb,
                &mut state.hb,
                hidden_dim,
                dim,
            );
            matmul(
                &weights.w3[ffn_offset..ffn_offset + hidden_dim * dim],
                &state.xb,
                &mut state.hb2,
                hidden_dim,
                dim,
            );
            for i in 0..hidden_dim {
                state.hb[i] = silu(state.hb[i]) * state.hb2[i];
            }

            matmul(
                &weights.w2[w2_offset..w2_offset + dim * hidden_dim],
                &state.hb,
                &mut state.xb,
                dim,
                hidden_dim,
            );
            for i in 0..dim {
                state.x[i] += state.xb[i];
            }
        }

        rmsnorm(&mut state.xb, &state.x, &weights.rms_final_weight);
        matmul(
            weights.classifier_weights(),
            &state.xb,
            &mut state.logits,
            config.vocab_size,
            dim,
        );

        Ok(&state.logits)
    }
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

impl fmt::Display for TransformerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidToken { token, vocab_size } => {
                write!(f, "token {token} is out of range for vocab size {vocab_size}")
            }
            Self::InvalidPosition { position, seq_len } => {
                write!(f, "position {position} is out of range for sequence length {seq_len}")
            }
            Self::InvalidWeights(reason) => write!(f, "invalid transformer weights: {reason}"),
            Self::WeightLayout(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for TransformerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::WeightLayout(err) => Some(err),
            Self::InvalidToken { .. }
            | Self::InvalidPosition { .. }
            | Self::InvalidWeights(_) => None,
        }
    }
}

fn validate_weight_shapes(config: &Config, weights: &Weights) -> Result<(), TransformerError> {
    let counts = WeightCounts::from_config(config).map_err(TransformerError::WeightLayout)?;

    check_len(
        "token_embedding_table",
        weights.token_embedding_table.len(),
        counts.token_embedding_table,
    )?;
    check_len("rms_att_weight", weights.rms_att_weight.len(), counts.rms_att_weight)?;
    check_len("wq", weights.wq.len(), counts.wq)?;
    check_len("wk", weights.wk.len(), counts.wk)?;
    check_len("wv", weights.wv.len(), counts.wv)?;
    check_len("wo", weights.wo.len(), counts.wo)?;
    check_len("rms_ffn_weight", weights.rms_ffn_weight.len(), counts.rms_ffn_weight)?;
    check_len("w1", weights.w1.len(), counts.w1)?;
    check_len("w2", weights.w2.len(), counts.w2)?;
    check_len("w3", weights.w3.len(), counts.w3)?;
    check_len(
        "rms_final_weight",
        weights.rms_final_weight.len(),
        counts.rms_final_weight,
    )?;
    check_len(
        "freq_cis_real",
        weights.freq_cis_real.len(),
        counts.freq_cis_real,
    )?;
    check_len(
        "freq_cis_imag",
        weights.freq_cis_imag.len(),
        counts.freq_cis_imag,
    )?;
    if let Some(wcls) = &weights.wcls {
        check_len("wcls", wcls.len(), counts.wcls)?;
    }

    Ok(())
}

fn check_len(name: &'static str, actual: usize, expected: usize) -> Result<(), TransformerError> {
    if actual != expected {
        return Err(TransformerError::InvalidWeights(match name {
            "token_embedding_table" => "token_embedding_table length mismatch",
            "rms_att_weight" => "rms_att_weight length mismatch",
            "wq" => "wq length mismatch",
            "wk" => "wk length mismatch",
            "wv" => "wv length mismatch",
            "wo" => "wo length mismatch",
            "rms_ffn_weight" => "rms_ffn_weight length mismatch",
            "w1" => "w1 length mismatch",
            "w2" => "w2 length mismatch",
            "w3" => "w3 length mismatch",
            "rms_final_weight" => "rms_final_weight length mismatch",
            "freq_cis_real" => "freq_cis_real length mismatch",
            "freq_cis_imag" => "freq_cis_imag length mismatch",
            "wcls" => "wcls length mismatch",
            _ => "weight length mismatch",
        }));
    }
    Ok(())
}

fn rmsnorm(out: &mut [f32], input: &[f32], weight: &[f32]) {
    let mut sum_sq = 0.0;
    for &value in input {
        sum_sq += value * value;
    }
    let rms = (sum_sq / input.len() as f32 + RMS_NORM_EPS).sqrt();
    let scale = 1.0 / rms;
    for i in 0..input.len() {
        out[i] = input[i] * scale * weight[i];
    }
}

fn attention(state: &mut RunState, config: &Config, kv_mul: usize, layer: usize, position: usize) {
    let head_dim = config.head_dim();
    let scale = 1.0 / (head_dim as f32).sqrt();

    state.xb.fill(0.0);
    for head in 0..config.n_heads {
        let q_start = head * head_dim;
        let att_offset = head * config.seq_len;
        let kv_head = head / kv_mul;

        for timestep in 0..=position {
            let key = cached_k(state, config, layer, timestep, kv_head);
            let mut score = dot(&state.q[q_start..q_start + head_dim], key) * scale;
            if !score.is_finite() {
                score = f32::NEG_INFINITY;
            }
            state.att[att_offset + timestep] = score;
        }

        softmax_in_place(&mut state.att[att_offset..att_offset + position + 1]);

        let out_start = head * head_dim;
        for i in 0..head_dim {
            state.xb[out_start + i] = 0.0;
        }
        for timestep in 0..=position {
            let weight = state.att[att_offset + timestep];
            let value_base =
                (layer * config.seq_len + timestep) * config.kv_dim() + kv_head * head_dim;
            for i in 0..head_dim {
                state.xb[out_start + i] += weight * state.value_cache[value_base + i];
            }
        }
    }
}

fn cache_kv(state: &mut RunState, config: &Config, layer: usize, position: usize) {
    let kv_dim = config.kv_dim();
    let base = (layer * config.seq_len + position) * kv_dim;
    state.key_cache[base..base + kv_dim].copy_from_slice(&state.k[..kv_dim]);
    state.value_cache[base..base + kv_dim].copy_from_slice(&state.v[..kv_dim]);
}

fn cached_k<'a>(
    state: &'a RunState,
    config: &Config,
    layer: usize,
    position: usize,
    kv_head: usize,
) -> &'a [f32] {
    let head_dim = config.head_dim();
    let kv_dim = config.kv_dim();
    let base = (layer * config.seq_len + position) * kv_dim + kv_head * head_dim;
    &state.key_cache[base..base + head_dim]
}

fn matmul(weight: &[f32], input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(weight.len(), rows * cols);
    debug_assert_eq!(input.len(), cols);
    debug_assert_eq!(out.len(), rows);

    for row in 0..rows {
        let w = &weight[row * cols..(row + 1) * cols];
        out[row] = dot(w, input);
    }
}

fn dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

fn apply_rope(
    vector: &mut [f32],
    freq_real: &[f32],
    freq_imag: &[f32],
    n_heads: usize,
    head_dim: usize,
) {
    for head in 0..n_heads {
        let head_slice = &mut vector[head * head_dim..(head + 1) * head_dim];
        for i in 0..head_dim / 2 {
            let even = 2 * i;
            let odd = even + 1;
            let real = freq_real[i];
            let imag = freq_imag[i];
            let x0 = head_slice[even];
            let x1 = head_slice[odd];
            head_slice[even] = x0 * real - x1 * imag;
            head_slice[odd] = x0 * imag + x1 * real;
        }
    }
}

fn softmax_in_place(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }

    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }

    if sum == 0.0 || !sum.is_finite() {
        let uniform = 1.0 / values.len() as f32;
        values.fill(uniform);
        return;
    }

    for value in values {
        *value /= sum;
    }
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::{Transformer, TransformerError};
    use crate::config::Config;
    use crate::weights::{WeightCounts, Weights};

    fn test_config() -> Config {
        Config {
            dim: 4,
            hidden_dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            vocab_size: 3,
            seq_len: 4,
        }
    }

    fn zero_weights(config: &Config) -> Weights {
        let counts = WeightCounts::from_config(config).expect("counts should compute");
        Weights {
            token_embedding_table: vec![0.0; counts.token_embedding_table],
            rms_att_weight: vec![0.0; counts.rms_att_weight],
            wq: vec![0.0; counts.wq],
            wk: vec![0.0; counts.wk],
            wv: vec![0.0; counts.wv],
            wo: vec![0.0; counts.wo],
            rms_ffn_weight: vec![0.0; counts.rms_ffn_weight],
            w1: vec![0.0; counts.w1],
            w2: vec![0.0; counts.w2],
            w3: vec![0.0; counts.w3],
            rms_final_weight: vec![0.0; counts.rms_final_weight],
            freq_cis_real: vec![0.0; counts.freq_cis_real],
            freq_cis_imag: vec![0.0; counts.freq_cis_imag],
            wcls: Some(vec![0.0; counts.wcls]),
        }
    }

    fn simple_weights(config: &Config) -> Weights {
        let counts = WeightCounts::from_config(config).expect("counts should compute");
        let mut weights = zero_weights(config);
        weights.token_embedding_table[..config.dim].copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        weights.rms_att_weight.fill(1.0);
        weights.rms_ffn_weight.fill(1.0);
        weights.rms_final_weight.fill(1.0);
        weights.freq_cis_real.fill(1.0);
        weights.freq_cis_imag.fill(0.0);
        weights.wk[..config.kv_dim() * config.dim].copy_from_slice(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
        ]);
        weights.wv[..config.kv_dim() * config.dim].copy_from_slice(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
        ]);
        weights.wcls = Some(vec![0.0; counts.wcls]);
        weights
    }

    #[test]
    fn zero_weights_produce_zero_logits() {
        let config = test_config();
        let weights = zero_weights(&config);
        let mut transformer = Transformer::new(config.clone(), weights).expect("transformer");

        let logits = transformer.forward(0, 0).expect("forward should work");
        assert_eq!(logits, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn forward_updates_kv_cache() {
        let config = test_config();
        let weights = simple_weights(&config);
        let mut transformer = Transformer::new(config.clone(), weights).expect("transformer");

        transformer.forward(0, 0).expect("forward should work");

// @trace-pilot 4ad8b2c1ac45cf708430b9574ecae2a1dc28081b
        assert_eq!(&transformer.state().key_cache[..2], &[1.999_96, 0.0]);
        assert_eq!(&transformer.state().value_cache[..2], &[1.999_96, 0.0]);
    }

    #[test]
    fn rejects_invalid_token_and_position() {
        let config = test_config();
        let weights = zero_weights(&config);
        let mut transformer = Transformer::new(config.clone(), weights).expect("transformer");

        let err = transformer.forward(99, 0).expect_err("should fail");
        assert!(matches!(err, TransformerError::InvalidToken { .. }));

        let err = transformer.forward(0, config.seq_len).expect_err("should fail");
        assert!(matches!(err, TransformerError::InvalidPosition { .. }));
    }
}
