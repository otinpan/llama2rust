// @trace-pilot 72c12de8a622ad27ab5603fca7dfc63366efe061
use std::fmt;

#[derive(Debug, Clone)]
pub struct Sampler {
    vocab_size: usize,
    temperature: f32,
    top_p: f32,
    rng_state: u64,
    probabilities: Vec<f32>,
    candidates: Vec<Candidate>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    token_id: u32,
    probability: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampleResult {
    pub token_id: u32,
    pub probabilities: Vec<f32>,
}

#[derive(Debug)]
pub enum SamplerError {
    InvalidConfig(&'static str),
    LogitCountMismatch { expected: usize, actual: usize },
    EmptyLogits,
}

impl Sampler {
    pub fn new(
        vocab_size: usize,
        temperature: f32,
        top_p: f32,
        seed: u64,
    ) -> Result<Self, SamplerError> {
        if vocab_size == 0 {
            return Err(SamplerError::InvalidConfig(
                "vocab_size must be greater than zero",
            ));
        }

        if temperature < 0.0 {
            return Err(SamplerError::InvalidConfig(
                "temperature must be greater than or equal to zero",
            ));
        }

        if !(0.0..=1.0).contains(&top_p) {
            return Err(SamplerError::InvalidConfig(
                "top_p must be between 0.0 and 1.0",
            ));
        }

        Ok(Self {
            vocab_size,
            temperature,
            top_p,
            rng_state: seed.max(1),
            probabilities: vec![0.0; vocab_size],
            candidates: Vec::with_capacity(vocab_size),
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    pub fn top_p(&self) -> f32 {
        self.top_p
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<u32, SamplerError> {
        Ok(self.sample_with_probs(logits)?.token_id)
    }

    pub fn sample_with_probs(&mut self, logits: &[f32]) -> Result<SampleResult, SamplerError> {
        self.validate_logits(logits)?;

        if self.temperature == 0.0 {
            let token_id = greedy_sample(logits)?;
            return Ok(SampleResult {
                token_id,
                probabilities: self.greedy_probabilities(logits)?,
            });
        }

        self.softmax(logits)?;
        let token_id = if self.top_p <= 0.0 || self.top_p >= 1.0 {
            self.sample_from_distribution()
        } else {
            self.sample_top_p()
        };

        Ok(SampleResult {
            token_id,
            probabilities: self.probabilities.clone(),
        })
    }

    fn validate_logits(&self, logits: &[f32]) -> Result<(), SamplerError> {
        if logits.is_empty() {
            return Err(SamplerError::EmptyLogits);
        }

        if logits.len() != self.vocab_size {
            return Err(SamplerError::LogitCountMismatch {
                expected: self.vocab_size,
                actual: logits.len(),
            });
        }

        Ok(())
    }

    fn greedy_probabilities(&self, logits: &[f32]) -> Result<Vec<f32>, SamplerError> {
        let token_id = greedy_sample(logits)? as usize;
        let mut probabilities = vec![0.0; self.vocab_size];
        probabilities[token_id] = 1.0;
        Ok(probabilities)
    }

    fn softmax(&mut self, logits: &[f32]) -> Result<(), SamplerError> {
        self.validate_logits(logits)?;

        let inv_temperature = 1.0 / self.temperature;
        let max_logit = logits
            .iter()
            .map(|logit| logit * inv_temperature)
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0;
        for (probability, logit) in self.probabilities.iter_mut().zip(logits.iter().copied()) {
            let adjusted = (logit * inv_temperature - max_logit).exp();
            *probability = adjusted;
            sum += adjusted;
        }

        if sum == 0.0 || !sum.is_finite() {
            let uniform = 1.0 / self.vocab_size as f32;
            self.probabilities.fill(uniform);
            return Ok(());
        }

        for probability in &mut self.probabilities {
            *probability /= sum;
        }

        Ok(())
    }

    fn sample_from_distribution(&mut self) -> u32 {
        let threshold = self.next_f32();
        let mut cumulative = 0.0;
        for (token_id, probability) in self.probabilities.iter().copied().enumerate() {
            cumulative += probability;
            if threshold <= cumulative || token_id + 1 == self.vocab_size {
                return token_id as u32;
            }
        }

        (self.vocab_size - 1) as u32
    }

    fn sample_top_p(&mut self) -> u32 {
        self.candidates.clear();
        self.candidates
            .extend(self.probabilities.iter().copied().enumerate().map(
                |(token_id, probability)| Candidate {
                    token_id: token_id as u32,
                    probability,
                },
            ));
        self.candidates.sort_by(|lhs, rhs| {
            rhs.probability
                .partial_cmp(&lhs.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cutoff = 0;
        let mut cumulative = 0.0;
        for candidate in &self.candidates {
            cumulative += candidate.probability;
            cutoff += 1;
            if cumulative >= self.top_p {
                break;
            }
        }

        let cutoff = cutoff.max(1);
        let total = self.candidates[..cutoff]
            .iter()
            .map(|candidate| candidate.probability)
            .sum::<f32>();
        if total <= 0.0 || !total.is_finite() {
// @trace-pilot 72c12de8a622ad27ab5603fca7dfc63366efe061
            return self.candidates[0].token_id;
        }

        let threshold = self.next_f32() * total;
        let mut cumulative = 0.0;
// @trace-pilot 72c12de8a622ad27ab5603fca7dfc63366efe061
        for candidate in &self.candidates[..cutoff] {
            cumulative += candidate.probability;
            if threshold <= cumulative {
                return candidate.token_id;
            }
        }

// @trace-pilot 72c12de8a622ad27ab5603fca7dfc63366efe061
        self.candidates[cutoff - 1].token_id
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f32(&mut self) -> f32 {
        let value = self.next_u64() >> 40;
        (value as f32) / ((1_u32 << 24) as f32)
    }
}

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(reason) => write!(f, "invalid sampler config: {reason}"),
            Self::LogitCountMismatch { expected, actual } => {
                write!(f, "logit count mismatch: expected {expected}, got {actual}")
            }
            Self::EmptyLogits => write!(f, "logits must not be empty"),
        }
    }
}

impl std::error::Error for SamplerError {}

fn greedy_sample(logits: &[f32]) -> Result<u32, SamplerError> {
    let (index, _) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.partial_cmp(&rhs.1).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or(SamplerError::EmptyLogits)?;
    Ok(index as u32)
}

#[cfg(test)]
mod tests {
    use super::{SampleResult, Sampler, SamplerError};

    fn probabilities_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected) {
            assert!((lhs - rhs).abs() < 1.0e-5, "lhs={lhs}, rhs={rhs}");
        }
    }

    #[test]
    fn greedy_sampling_selects_max_logit() {
        let mut sampler = Sampler::new(4, 0.0, 1.0, 123).expect("sampler should build");
        let result = sampler
            .sample_with_probs(&[0.1, 2.0, 0.3, -1.0])
            .expect("sample should succeed");

        assert_eq!(result.token_id, 1);
        probabilities_close(&result.probabilities, &[0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn softmax_normalizes_distribution() {
        let mut sampler = Sampler::new(3, 1.0, 1.0, 1).expect("sampler should build");
        let SampleResult { probabilities, .. } = sampler
            .sample_with_probs(&[1.0, 1.0, 1.0])
            .expect("sample should succeed");

        probabilities_close(&probabilities, &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    }

    #[test]
    fn top_p_limits_candidates_to_nucleus() {
        let mut sampler = Sampler::new(4, 1.0, 0.75, 1).expect("sampler should build");
        for _ in 0..32 {
            let token = sampler
                .sample(&[4.0, 3.0, 1.0, -2.0])
                .expect("sample should succeed");
            assert!(token == 0 || token == 1);
        }
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let logits = [1.0, 0.9, 0.2, -1.0];
        let mut lhs = Sampler::new(4, 0.8, 1.0, 42).expect("sampler should build");
        let mut rhs = Sampler::new(4, 0.8, 1.0, 42).expect("sampler should build");

        let left = (0..10)
            .map(|_| lhs.sample(&logits).expect("sample should succeed"))
            .collect::<Vec<_>>();
        let right = (0..10)
            .map(|_| rhs.sample(&logits).expect("sample should succeed"))
            .collect::<Vec<_>>();

        assert_eq!(left, right);
    }

    #[test]
    fn rejects_invalid_configuration() {
        let err = Sampler::new(0, 1.0, 1.0, 1).expect_err("sampler should fail");
        assert!(matches!(err, SamplerError::InvalidConfig(_)));

        let err = Sampler::new(8, -0.1, 1.0, 1).expect_err("sampler should fail");
        assert!(matches!(err, SamplerError::InvalidConfig(_)));

        let err = Sampler::new(8, 1.0, 1.5, 1).expect_err("sampler should fail");
        assert!(matches!(err, SamplerError::InvalidConfig(_)));
    }

    #[test]
    fn rejects_logit_count_mismatch() {
        let mut sampler = Sampler::new(4, 1.0, 1.0, 1).expect("sampler should build");
        let err = sampler.sample(&[1.0, 2.0]).expect_err("sample should fail");
        assert!(matches!(
            err,
            SamplerError::LogitCountMismatch {
                expected: 4,
                actual: 2
            }
        ));
    }
}
