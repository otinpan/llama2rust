// @trace-pilot 78c1b1a6b0c7b5d835f81a2de7b17fb68b6aa045
use std::fmt;

use crate::sampler::{Sampler, SamplerError};
use crate::tokenizer::{Tokenizer, TokenizerError};
use crate::transformer::{Transformer, TransformerError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateOptions {
    pub prompt: String,
    pub steps: usize,
    pub add_bos: bool,
    pub stop_at_eos: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateResult {
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub generated_text: String,
    pub steps_taken: usize,
    pub stopped_by_eos: bool,
}

#[derive(Debug)]
pub enum GenerateError {
    EmptyPrompt,
    PromptTooLong {
        prompt_len: usize,
        seq_len: usize,
    },
    Tokenizer(TokenizerError),
    Sampler(SamplerError),
    Transformer(TransformerError),
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            steps: 128,
            add_bos: true,
            stop_at_eos: true,
        }
    }
}

pub fn generate(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    options: &GenerateOptions,
) -> Result<GenerateResult, GenerateError> {
    let mut prompt_tokens = tokenizer
        .encode(&options.prompt, options.add_bos, false)
        .map_err(GenerateError::Tokenizer)?;

    if prompt_tokens.is_empty() {
// @trace-pilot 78c1b1a6b0c7b5d835f81a2de7b17fb68b6aa045
        if options.add_bos {
            if let Some(bos) = tokenizer.bos_token() {
                prompt_tokens.push(bos);
            } else {
                return Err(GenerateError::EmptyPrompt);
            }
        } else {
            return Err(GenerateError::EmptyPrompt);
        }
    }

    if prompt_tokens.len() > transformer.config().seq_len {
        return Err(GenerateError::PromptTooLong {
            prompt_len: prompt_tokens.len(),
            seq_len: transformer.config().seq_len,
        });
    }

    let mut current_position = 0usize;
    let mut logits = transformer
        .forward(prompt_tokens[0], current_position)
        .map_err(GenerateError::Transformer)?
        .to_vec();

    for &token in &prompt_tokens[1..] {
        current_position += 1;
        logits = transformer
            .forward(token, current_position)
            .map_err(GenerateError::Transformer)?
            .to_vec();
    }

    let mut generated_tokens = Vec::new();
    let mut generated_text = String::new();
    let eos_token = tokenizer.eos_token();
    let max_steps = options
        .steps
        .min(transformer.config().seq_len.saturating_sub(prompt_tokens.len()));
    let mut stopped_by_eos = false;

    for _ in 0..max_steps {
        let next_token = sampler.sample(&logits).map_err(GenerateError::Sampler)?;
        if options.stop_at_eos && eos_token == Some(next_token) {
            stopped_by_eos = true;
            break;
        }

        generated_text.push_str(
            &tokenizer
                .decode(next_token)
                .map_err(GenerateError::Tokenizer)?,
        );
        generated_tokens.push(next_token);

        current_position += 1;
        logits = transformer
            .forward(next_token, current_position)
            .map_err(GenerateError::Transformer)?
            .to_vec();
    }

    let steps_taken = generated_tokens.len();
    Ok(GenerateResult {
        prompt_tokens,
        generated_tokens,
        generated_text,
        steps_taken,
        stopped_by_eos,
    })
}

impl fmt::Display for GenerateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPrompt => write!(f, "prompt produced no tokens and tokenizer has no BOS token"),
            Self::PromptTooLong {
                prompt_len,
                seq_len,
            } => write!(
                f,
                "prompt is too long for the model context: {prompt_len} tokens > {seq_len}"
            ),
            Self::Tokenizer(err) => write!(f, "{err}"),
            Self::Sampler(err) => write!(f, "{err}"),
            Self::Transformer(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for GenerateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tokenizer(err) => Some(err),
            Self::Sampler(err) => Some(err),
            Self::Transformer(err) => Some(err),
            Self::EmptyPrompt | Self::PromptTooLong { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{generate, GenerateError, GenerateOptions};
    use crate::config::Config;
    use crate::sampler::Sampler;
    use crate::tokenizer::Tokenizer;
    use crate::transformer::Transformer;
    use crate::weights::{WeightCounts, Weights};

    fn tokenizer_bytes(max_token_length: u32, entries: &[(f32, &[u8])]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&max_token_length.to_le_bytes());
        for (score, token) in entries {
            bytes.extend_from_slice(&score.to_le_bytes());
            bytes.extend_from_slice(&(token.len() as i32).to_le_bytes());
            bytes.extend_from_slice(token);
        }
        bytes
    }

    fn test_tokenizer() -> Tokenizer {
        let entries = vec![
            (0.0, b"<unk>" as &[u8]),
            (0.0, b"<s>"),
            (0.0, b"</s>"),
            (1.0, b"a"),
        ];
        let bytes = tokenizer_bytes(5, &entries);
        let mut reader = bytes.as_slice();
        Tokenizer::from_reader(entries.len(), &mut reader).expect("tokenizer should load")
    }

    fn test_config() -> Config {
        Config {
            dim: 4,
            hidden_dim: 8,
            n_layers: 1,
            n_heads: 2,
            n_kv_heads: 1,
            vocab_size: 4,
            seq_len: 8,
        }
    }

    fn test_transformer() -> Transformer {
        let config = test_config();
        let counts = WeightCounts::from_config(&config).expect("counts should compute");
        let mut weights = Weights {
            token_embedding_table: vec![0.0; counts.token_embedding_table],
            rms_att_weight: vec![1.0; counts.rms_att_weight],
            wq: vec![0.0; counts.wq],
            wk: vec![0.0; counts.wk],
            wv: vec![0.0; counts.wv],
            wo: vec![0.0; counts.wo],
            rms_ffn_weight: vec![1.0; counts.rms_ffn_weight],
            w1: vec![0.0; counts.w1],
            w2: vec![0.0; counts.w2],
            w3: vec![0.0; counts.w3],
            rms_final_weight: vec![1.0; counts.rms_final_weight],
            freq_cis_real: vec![1.0; counts.freq_cis_real],
            freq_cis_imag: vec![0.0; counts.freq_cis_imag],
            wcls: Some(vec![0.0; counts.wcls]),
        };

        weights.token_embedding_table[4 * 3] = 1.0;
        if let Some(wcls) = &mut weights.wcls {
            wcls[3 * config.dim] = 1.0;
        }

        Transformer::new(config, weights).expect("transformer should build")
    }

    #[test]
    fn generates_tokens_from_prompt() {
        let tokenizer = test_tokenizer();
        let mut transformer = test_transformer();
        let mut sampler = Sampler::new(4, 0.0, 1.0, 1).expect("sampler should build");
        let options = GenerateOptions {
            prompt: "a".to_string(),
            steps: 2,
            add_bos: false,
            stop_at_eos: true,
        };

        let result = generate(&mut transformer, &tokenizer, &mut sampler, &options)
            .expect("generation should work");

        assert_eq!(result.prompt_tokens, vec![3]);
        assert_eq!(result.generated_tokens, vec![3, 3]);
        assert_eq!(result.generated_text, "aa");
        assert!(!result.stopped_by_eos);
    }

    #[test]
    fn stops_on_eos_before_appending_output() {
        let tokenizer = test_tokenizer();
        let config = test_config();
        let counts = WeightCounts::from_config(&config).expect("counts should compute");
        let mut weights = Weights {
            token_embedding_table: vec![0.0; counts.token_embedding_table],
            rms_att_weight: vec![1.0; counts.rms_att_weight],
            wq: vec![0.0; counts.wq],
            wk: vec![0.0; counts.wk],
            wv: vec![0.0; counts.wv],
            wo: vec![0.0; counts.wo],
            rms_ffn_weight: vec![1.0; counts.rms_ffn_weight],
            w1: vec![0.0; counts.w1],
            w2: vec![0.0; counts.w2],
            w3: vec![0.0; counts.w3],
            rms_final_weight: vec![1.0; counts.rms_final_weight],
            freq_cis_real: vec![1.0; counts.freq_cis_real],
            freq_cis_imag: vec![0.0; counts.freq_cis_imag],
            wcls: Some(vec![0.0; counts.wcls]),
        };
        weights.token_embedding_table[4 * 3] = 1.0;
        if let Some(wcls) = &mut weights.wcls {
            wcls[2 * config.dim] = 1.0;
        }

        let mut transformer = Transformer::new(config, weights).expect("transformer should build");
        let mut sampler = Sampler::new(4, 0.0, 1.0, 1).expect("sampler should build");
        let options = GenerateOptions {
            prompt: "a".to_string(),
            steps: 2,
            add_bos: false,
            stop_at_eos: true,
        };

        let result = generate(&mut transformer, &tokenizer, &mut sampler, &options)
            .expect("generation should work");

        assert!(result.generated_tokens.is_empty());
        assert!(result.generated_text.is_empty());
        assert!(result.stopped_by_eos);
    }

    #[test]
    fn rejects_empty_prompt_without_bos() {
        let tokenizer = test_tokenizer();
        let mut transformer = test_transformer();
        let mut sampler = Sampler::new(4, 0.0, 1.0, 1).expect("sampler should build");
        let options = GenerateOptions {
            prompt: String::new(),
            steps: 1,
            add_bos: false,
            stop_at_eos: true,
        };

        let err =
            generate(&mut transformer, &tokenizer, &mut sampler, &options).expect_err("should fail");
        assert!(matches!(err, GenerateError::EmptyPrompt));
    }
}
