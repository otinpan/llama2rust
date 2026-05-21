use crate::kernels::softmax;

#[derive(Debug, Clone, Copy, PartialEq)]
struct ProbIndex {
    prob: f32,
    index: usize,
}

#[derive(Debug, Clone)]
pub struct Sampler {
    vocab_size: usize,
    probindex: Vec<ProbIndex>,
    temperature: f32,
    topp: f32,
    rng_state: u64,
}

impl Sampler {
    pub fn new(vocab_size: usize, temperature: f32, topp: f32, rng_seed: u64) -> Self {
        Self {
            vocab_size,
            probindex: vec![
                ProbIndex {
                    prob: 0.0,
                    index: 0,
                };
                vocab_size
            ],
            temperature,
            topp,
            rng_state: rng_seed,
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> usize {
        assert_eq!(
            logits.len(),
            self.vocab_size,
            "logit length must match sampler vocab size"
        );

        if self.temperature == 0.0 {
            return sample_argmax(logits);
        }

        let mut probabilities = logits.to_vec();
        for value in &mut probabilities {
            *value /= self.temperature;
        }
        softmax(&mut probabilities);

        let coin = random_f32(&mut self.rng_state);
        if self.topp <= 0.0 || self.topp >= 1.0 {
            sample_mult(&probabilities, coin)
        } else {
            sample_topp(&probabilities, self.topp, &mut self.probindex, coin)
        }
    }
}

pub fn sample_argmax(probabilities: &[f32]) -> usize {
    probabilities
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("sample_argmax requires a non-empty slice")
}

fn sample_mult(probabilities: &[f32], coin: f32) -> usize {
    let mut cdf = 0.0_f32;
    for (index, probability) in probabilities.iter().copied().enumerate() {
        cdf += probability;
        if coin < cdf {
            return index;
        }
    }
    probabilities.len() - 1
}

fn sample_topp(
    probabilities: &[f32],
    topp: f32,
    probindex: &mut [ProbIndex],
    coin: f32,
) -> usize {
    let cutoff = (1.0 - topp) / (probabilities.len() - 1) as f32;
    let mut n0 = 0;
    for (index, probability) in probabilities.iter().copied().enumerate() {
        if probability >= cutoff {
            probindex[n0] = ProbIndex {
                prob: probability,
                index,
            };
            n0 += 1;
        }
    }
    probindex[..n0].sort_by(|left, right| right.prob.total_cmp(&left.prob));

    let mut cumulative_prob = 0.0_f32;
    let mut last_idx = n0.saturating_sub(1);
    for (index, item) in probindex[..n0].iter().enumerate() {
        cumulative_prob += item.prob;
        if cumulative_prob > topp {
            last_idx = index;
            break;
        }
    }

    let r = coin * cumulative_prob;
    let mut cdf = 0.0_f32;
    for item in &probindex[..=last_idx] {
        cdf += item.prob;
        if r < cdf {
            return item.index;
        }
    }
    probindex[last_idx].index
}

fn random_u32(state: &mut u64) -> u32 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    ((*state).wrapping_mul(0x2545_F491_4F6C_DD1D_u64) >> 32) as u32
}

fn random_f32(state: &mut u64) -> f32 {
    (random_u32(state) >> 8) as f32 / 16_777_216.0
}

#[cfg(test)]
mod tests {
    use super::{sample_argmax, sample_mult, sample_topp, Sampler};

    #[test]
    fn argmax_matches_highest_logit() {
        assert_eq!(sample_argmax(&[0.1, 0.8, 0.3]), 1);
    }

    #[test]
    fn multinomial_uses_cdf() {
        assert_eq!(sample_mult(&[0.2, 0.3, 0.5], 0.0), 0);
        assert_eq!(sample_mult(&[0.2, 0.3, 0.5], 0.21), 1);
        assert_eq!(sample_mult(&[0.2, 0.3, 0.5], 0.99), 2);
    }

    #[test]
    fn topp_sampling_stays_within_truncated_set() {
        let probabilities = [0.6, 0.25, 0.1, 0.05];
        let mut probindex = vec![super::ProbIndex { prob: 0.0, index: 0 }; 4];
        let sampled = sample_topp(&probabilities, 0.7, &mut probindex, 0.95);
        assert!(sampled == 0 || sampled == 1);
    }

    #[test]
    fn sampler_is_deterministic_with_fixed_seed() {
        let logits = [1.0, 2.0, 3.0];
        let mut left = Sampler::new(3, 1.0, 0.9, 1234);
        let mut right = Sampler::new(3, 1.0, 0.9, 1234);

        assert_eq!(left.sample(&logits), right.sample(&logits));
        assert_eq!(left.sample(&logits), right.sample(&logits));
    }

    #[test]
    fn zero_temperature_falls_back_to_argmax() {
        let logits = [1.0, 5.0, 2.0];
        let mut sampler = Sampler::new(3, 0.0, 0.9, 1);
        assert_eq!(sampler.sample(&logits), 1);
    }
}
