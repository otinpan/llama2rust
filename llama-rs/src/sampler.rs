// @trace-pilot 98257553dbb6647925fa2cc6bf95febb5a1373f2
// Sampler;

#[derive(Debug,Clone,Copy)]
pub struct ProbIndex{
    pub prob: f32,
    pub index: usize,
}

#[derive(Debug)]
pub struct Sampler{
    pub vocab_size: usize,
    pub prob_index: Vec<ProbIndex>,
    pub temperature: f32,
    pub topp: f32,
    pub rng_state: u64,
}

impl Sampler{
    // @trace-pilot a954bf9684037b17f19305e5ef58919a83204148
    // void build_sampler
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, rng_seed: u64) -> Self{
        let prob_index=vec![
            ProbIndex{
                prob: 0.0,
                index: 0,
            };
            vocab_size
        ];

        Self{
            vocab_size,
            prob_index,
            temperature,
            topp,
            rng_state: rng_seed,
        }
    }
    // @trace-pilot 42687873feda2ba1bbaebb14d293c81961cbce0a
    // int sample(
// @trace-pilot cf1d4e9c8e3e43600d4f17726142048fe3c84c2b
    pub fn sample(&mut self, logits: Vec<f32>) -> u32 {
        if logits.is_empty() {
            return 0;
        }

// @trace-pilot cf1d4e9c8e3e43600d4f17726142048fe3c84c2b
        if self.temperature == 0.0 {
            return argmax(&logits) as u32;
        }

        let mut probs = logits;
        for logit in &mut probs {
            *logit /= self.temperature;
        }
        softmax(&mut probs);

// @trace-pilot cf1d4e9c8e3e43600d4f17726142048fe3c84c2b
        let coin = random_f32(&mut self.rng_state);
        // @trace-pilot e32a8133022df4cb0666303cbb44d3ac59a8bf33
        // 次のtokenをどう選ぶか
        if self.topp <= 0.0 || self.topp >= 1.0 {
            sample_mult(&probs, coin) as u32
        } else {
            sample_topp(
                &probs,
                self.topp,
                &mut self.prob_index,
                coin,
            ) as u32
        }
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = values[0];

    for (idx, &value) in values.iter().enumerate().skip(1) {
        if value > best_val {
            best_val = value;
            best_idx = idx;
        }
    }

    best_idx
}

fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;

    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }

    if sum > 0.0 {
        for value in values.iter_mut() {
            *value /= sum;
        }
    }
}

// @trace-pilot cf1d4e9c8e3e43600d4f17726142048fe3c84c2b
// 累積確率でのサンプリング
// @trace-pilot 58a108dd7499a43c0fee4080ec469db86b7e25fa
// int sample_mult(
//
// @trace-pilot 0589b3ec84682af53b17ee64a1b9a080bb1b0c19
// sample index from probabilities (they must sum to 1!)
// coin is a random number in [0, 1), usually from random_f32()
fn sample_mult(probabilities: &[f32], coin: f32) -> usize {
    let mut cdf = 0.0f32;
    for (i, &prob) in probabilities.iter().enumerate() {
        cdf += prob;
        if coin < cdf {
            return i;
        }
    }
    probabilities.len().saturating_sub(1)
}

// @trace-pilot ff9221a779b9f533158298808ca95d26ef5020f3
// int sample_topp(f
//
// @trace-pilot 72009805c6cf931ffe3dcc16b0bf3b950017c838
// top-p sampling (or "nucleus sampling") samples from the smallest set of
// tokens that exceed probability topp. This way we never sample tokens that
// have very low probabilities and are less likely to go "off the rails".
// coin is a random number in [0, 1), usually from random_f32()
fn sample_topp(
    probabilities: &[f32],
    topp: f32,
    prob_index: &mut [ProbIndex],
    coin: f32,
) -> usize {
    let n = probabilities.len();
    let cutoff = (1.0f32 - topp) / (n as f32 - 1.0);

    let mut n0 = 0usize;
    for (i, &prob) in probabilities.iter().enumerate() {
        if prob >= cutoff {
            prob_index[n0].index = i;
            prob_index[n0].prob = prob;
            n0 += 1;
        }
    }

    prob_index[..n0].sort_by(|a, b| b.prob.total_cmp(&a.prob));

    let mut cumulative_prob = 0.0f32;
    let mut last_idx = n0.saturating_sub(1);
    for (i, item) in prob_index[..n0].iter().enumerate() {
        cumulative_prob += item.prob;
        if cumulative_prob > topp {
            last_idx = i;
            break;
        }
    }

    let r = coin * cumulative_prob;
    let mut cdf = 0.0f32;
    for item in &prob_index[..=last_idx] {
        cdf += item.prob;
        if r < cdf {
            return item.index;
        }
    }

    prob_index[last_idx].index
}

fn random_u32(state: &mut u64) -> u32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ((*state).wrapping_mul(0x2545_F491_4F6C_DD1D_u64) >> 32) as u32
}

fn random_f32(state: &mut u64) -> f32 {
    (random_u32(state) >> 8) as f32 / 16_777_216.0f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_returns_index_of_largest_value() {
        let values = [0.1, 2.5, 1.5, 2.4];
        assert_eq!(argmax(&values), 1);
    }

    #[test]
    fn sample_mult_uses_cumulative_probability() {
        let probabilities = [0.2, 0.5, 0.3];

        assert_eq!(sample_mult(&probabilities, 0.10), 0);
        assert_eq!(sample_mult(&probabilities, 0.40), 1);
        assert_eq!(sample_mult(&probabilities, 0.95), 2);
    }

    #[test]
    fn sample_topp_samples_only_from_nucleus() {
        let probabilities = [0.5, 0.3, 0.15, 0.05];
        let mut prob_index = vec![
            ProbIndex { prob: 0.0, index: 0 };
            probabilities.len()
        ];

        // topp=0.75 keeps 0.5 and 0.3, so token 2 and 3 must never be selected.
        assert_eq!(sample_topp(&probabilities, 0.75, &mut prob_index, 0.10), 0);
        assert_eq!(sample_topp(&probabilities, 0.75, &mut prob_index, 0.90), 1);
    }

    #[test]
    fn sample_with_zero_temperature_is_greedy() {
        let mut sampler = Sampler::new(3, 0.0, 0.9, 123);
        let logits = vec![0.1, 3.0, 2.0];

        assert_eq!(sampler.sample(logits), 1);
    }

    #[test]
    fn random_f32_stays_in_unit_interval() {
        let mut state = 42u64;

        for _ in 0..32 {
            let value = random_f32(&mut state);
            assert!((0.0..1.0).contains(&value), "value out of range: {value}");
        }
    }
}
