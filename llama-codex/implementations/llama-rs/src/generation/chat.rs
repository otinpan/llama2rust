// @trace-pilot 6519512a6ba496cc5c73f776df356292b1113527
use std::fmt;

use crate::generation::generate::{generate, GenerateError, GenerateOptions, GenerateResult};
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;

const DEFAULT_SYSTEM_PROMPT: &str = "";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatOptions {
    pub system_prompt: Option<String>,
    pub user_prompt: String,
    pub history: Vec<ChatTurn>,
    pub steps: usize,
    pub stop_at_eos: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChatResult {
    pub rendered_prompt: String,
    pub assistant_response: String,
    pub generated_tokens: Vec<u32>,
    pub prompt_tokens: Vec<u32>,
    pub steps_taken: usize,
    pub stopped_by_eos: bool,
}

#[derive(Debug)]
pub enum ChatError {
    EmptyUserPrompt,
    HistoryEndsWithUser,
    OrphanAssistantTurn,
    Generate(GenerateError),
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            system_prompt: None,
            user_prompt: String::new(),
            history: Vec::new(),
            steps: 128,
            stop_at_eos: true,
        }
    }
}

pub fn chat(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    options: &ChatOptions,
) -> Result<ChatResult, ChatError> {
    if options.user_prompt.trim().is_empty() {
        return Err(ChatError::EmptyUserPrompt);
    }

    let rendered_prompt = render_chat_prompt(
        options.system_prompt.as_deref(),
        &options.history,
        &options.user_prompt,
    )?;

    let generate_options = GenerateOptions {
        prompt: rendered_prompt.clone(),
        steps: options.steps,
        add_bos: true,
        stop_at_eos: options.stop_at_eos,
    };

    let GenerateResult {
        prompt_tokens,
        generated_tokens,
        generated_text,
        steps_taken,
        stopped_by_eos,
    } = generate(transformer, tokenizer, sampler, &generate_options)
        .map_err(ChatError::Generate)?;

    Ok(ChatResult {
        rendered_prompt,
        assistant_response: generated_text,
        generated_tokens,
        prompt_tokens,
        steps_taken,
        stopped_by_eos,
    })
}

pub fn render_chat_prompt(
    system_prompt: Option<&str>,
    history: &[ChatTurn],
    user_prompt: &str,
) -> Result<String, ChatError> {
    if user_prompt.trim().is_empty() {
        return Err(ChatError::EmptyUserPrompt);
    }

    let mut rendered = String::new();
    let system_prompt = system_prompt.unwrap_or(DEFAULT_SYSTEM_PROMPT);
    let mut pending_user: Option<&str> = None;

    for turn in history {
        match turn.role {
            ChatRole::User => {
                if pending_user.is_some() {
                    return Err(ChatError::HistoryEndsWithUser);
                }
                pending_user = Some(turn.content.as_str());
            }
            ChatRole::Assistant => {
                let user_content = pending_user.take().ok_or(ChatError::OrphanAssistantTurn)?;
                if rendered.is_empty() {
                    rendered.push_str(&render_inst_block(Some(system_prompt), user_content));
                } else {
                    rendered.push_str(&render_inst_block(None, user_content));
                }
                rendered.push(' ');
                rendered.push_str(turn.content.trim());
                rendered.push(' ');
            }
        }
    }

    if pending_user.is_some() {
        return Err(ChatError::HistoryEndsWithUser);
    }

    if rendered.is_empty() {
        rendered.push_str(&render_inst_block(Some(system_prompt), user_prompt));
    } else {
        rendered.push_str(&render_inst_block(None, user_prompt));
    }

    Ok(rendered)
}

fn render_inst_block(system_prompt: Option<&str>, user_prompt: &str) -> String {
    match system_prompt {
        Some(system_prompt) if !system_prompt.trim().is_empty() => format!(
            "[INST]\n<<SYS>>\n{}\n<</SYS>>\n\n{}\n[/INST]",
            system_prompt.trim(),
            user_prompt.trim()
        ),
        _ => format!("[INST] {} [/INST]", user_prompt.trim()),
    }
}

impl fmt::Display for ChatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyUserPrompt => write!(f, "user prompt must not be empty"),
            Self::HistoryEndsWithUser => write!(
                f,
                "chat history must contain complete user/assistant pairs before the current user prompt"
            ),
            Self::OrphanAssistantTurn => {
                write!(f, "assistant turn appeared without a preceding user turn")
            }
            Self::Generate(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ChatError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Generate(err) => Some(err),
            Self::EmptyUserPrompt | Self::HistoryEndsWithUser | Self::OrphanAssistantTurn => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{chat, render_chat_prompt, ChatError, ChatOptions, ChatRole, ChatTurn};
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
            (1.0, b" "),
            (1.0, b"["),
            (1.0, b"]"),
            (1.0, b"/"),
            (1.0, b"I"),
            (1.0, b"N"),
            (1.0, b"S"),
            (1.0, b"T"),
            (1.0, b"<"),
            (1.0, b">"),
            (1.0, b"Y"),
            (1.0, b"\n"),
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
            vocab_size: 16,
            seq_len: 128,
        }
    }

    fn test_transformer(next_token: u32) -> Transformer {
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

// @trace-pilot 6519512a6ba496cc5c73f776df356292b1113527
        for token in 0..config.vocab_size {
            weights.token_embedding_table[token * config.dim] = 1.0;
        }
        if let Some(wcls) = &mut weights.wcls {
            wcls[next_token as usize * config.dim] = 1.0;
        }

        Transformer::new(config, weights).expect("transformer should build")
    }

    #[test]
    fn renders_system_prompt_template() {
        let prompt = render_chat_prompt(Some("system"), &[], "user").expect("prompt should render");
        assert_eq!(
            prompt,
            "[INST]\n<<SYS>>\nsystem\n<</SYS>>\n\nuser\n[/INST]"
        );
    }

    #[test]
    fn renders_history_and_current_user_prompt() {
        let history = vec![
            ChatTurn {
                role: ChatRole::User,
                content: "first".to_string(),
            },
            ChatTurn {
                role: ChatRole::Assistant,
                content: "reply".to_string(),
            },
        ];

        let prompt =
            render_chat_prompt(Some("system"), &history, "next").expect("prompt should render");
        assert_eq!(
            prompt,
            "[INST]\n<<SYS>>\nsystem\n<</SYS>>\n\nfirst\n[/INST] reply [INST] next [/INST]"
        );
    }

    #[test]
    fn rejects_invalid_history_shapes() {
        let err = render_chat_prompt(
            None,
            &[ChatTurn {
                role: ChatRole::Assistant,
                content: "reply".to_string(),
            }],
            "next",
        )
        .expect_err("prompt should fail");
        assert!(matches!(err, ChatError::OrphanAssistantTurn));

        let err = render_chat_prompt(
            None,
            &[ChatTurn {
                role: ChatRole::User,
                content: "first".to_string(),
            }],
            "next",
        )
        .expect_err("prompt should fail");
        assert!(matches!(err, ChatError::HistoryEndsWithUser));
    }

    #[test]
    fn chat_runs_generation_with_rendered_prompt() {
        let tokenizer = test_tokenizer();
        let mut transformer = test_transformer(3);
        let mut sampler = Sampler::new(16, 0.0, 1.0, 1).expect("sampler should build");
        let options = ChatOptions {
            system_prompt: None,
            user_prompt: "a".to_string(),
            history: Vec::new(),
            steps: 2,
            stop_at_eos: true,
        };

        let result =
            chat(&mut transformer, &tokenizer, &mut sampler, &options).expect("chat should work");

        assert_eq!(result.rendered_prompt, "[INST] a [/INST]");
        assert_eq!(result.assistant_response, "aa");
        assert_eq!(result.generated_tokens, vec![3, 3]);
    }
}
