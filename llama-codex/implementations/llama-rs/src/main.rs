mod config;
mod generation;
mod sampler;
mod tokenizer;
mod transformer;
mod weights;

use std::env;
use std::fmt;
use std::path::PathBuf;
use std::process;
use std::str::FromStr;

use generation::chat::{chat, ChatOptions};
use generation::generate::{generate, GenerateOptions};
use generation::mode::GenerationMode;
use sampler::Sampler;
use tokenizer::Tokenizer;
use transformer::Transformer;
use weights::Weights;

use crate::config::Config;

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        process::exit(1);
    }
}

fn run() -> Result<(), AppError> {
// @trace-pilot 6aab9f60c4dceb3631e613ec20c4ada5e9f0f138
    let cli = Cli::parse(env::args().skip(1)).map_err(AppError::Cli)?;

    let config = Config::from_model_file(&cli.model).map_err(AppError::Config)?;
    let weights = Weights::from_model_file(&cli.model, &config).map_err(AppError::Weights)?;
    let tokenizer = Tokenizer::from_file(&cli.tokenizer, config.vocab_size).map_err(AppError::Tokenizer)?;
    let mut sampler = Sampler::new(config.vocab_size, cli.temperature, cli.top_p, cli.seed)
        .map_err(AppError::Sampler)?;
    let mut transformer = Transformer::new(config, weights).map_err(AppError::Transformer)?;

    match cli.mode {
        GenerationMode::Generate => {
            let result = generate(
                &mut transformer,
                &tokenizer,
                &mut sampler,
                &GenerateOptions {
                    prompt: cli.input.unwrap_or_default(),
                    steps: cli.steps,
                    add_bos: true,
                    stop_at_eos: true,
                },
            )
            .map_err(AppError::Generate)?;
            print!("{}", result.generated_text);
        }
        GenerationMode::Chat => {
            let user_prompt = cli
                .input
                .filter(|value| !value.trim().is_empty())
                .ok_or(AppError::Cli(CliError::MissingValue(
                    "chat mode requires `-i` or `--input`",
                )))?;
            let result = chat(
                &mut transformer,
                &tokenizer,
                &mut sampler,
                &ChatOptions {
                    system_prompt: cli.system_prompt,
                    user_prompt,
                    history: Vec::new(),
                    steps: cli.steps,
                    stop_at_eos: true,
                },
            )
            .map_err(AppError::Chat)?;
            print!("{}", result.assistant_response);
        }
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
struct Cli {
    model: PathBuf,
    tokenizer: PathBuf,
    input: Option<String>,
    mode: GenerationMode,
    steps: usize,
    temperature: f32,
    top_p: f32,
    seed: u64,
    system_prompt: Option<String>,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            tokenizer: PathBuf::new(),
            input: None,
            mode: GenerationMode::Generate,
            steps: 128,
            temperature: 1.0,
            top_p: 0.9,
            seed: 42,
            system_prompt: None,
        }
    }
}

impl Cli {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, CliError> {
        let mut cli = Self::default();
        let mut args = args.into_iter();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => return Err(CliError::HelpRequested),
                "--model" => cli.model = PathBuf::from(next_value(&mut args, "--model")?),
                "-t" | "--tokenizer" => {
                    cli.tokenizer = PathBuf::from(next_value(&mut args, "--tokenizer")?)
                }
                "-i" | "--input" => cli.input = Some(next_value(&mut args, "--input")?),
                "-m" | "--mode" => {
                    cli.mode = GenerationMode::from_str(&next_value(&mut args, "--mode")?)
                        .map_err(CliError::Mode)?
                }
                "--steps" => cli.steps = parse_value(&next_value(&mut args, "--steps")?, "--steps")?,
                "--temp" | "--temperature" => {
                    cli.temperature =
                        parse_value(&next_value(&mut args, "--temperature")?, "--temperature")?
                }
                "--top-p" => cli.top_p = parse_value(&next_value(&mut args, "--top-p")?, "--top-p")?,
                "--seed" => cli.seed = parse_value(&next_value(&mut args, "--seed")?, "--seed")?,
                "--system" => cli.system_prompt = Some(next_value(&mut args, "--system")?),
                other if other.starts_with('-') => {
                    return Err(CliError::UnknownFlag(other.to_string()))
                }
                other => {
                    if cli.model.as_os_str().is_empty() {
                        cli.model = PathBuf::from(other);
                    } else if cli.tokenizer.as_os_str().is_empty() {
                        cli.tokenizer = PathBuf::from(other);
                    } else if cli.input.is_none() {
                        cli.input = Some(other.to_string());
                    } else {
                        return Err(CliError::UnexpectedArgument(other.to_string()));
                    }
                }
            }
        }

        if cli.model.as_os_str().is_empty() {
            return Err(CliError::MissingValue(
                "missing model path; pass it positionally or with `--model`",
            ));
        }

        if cli.tokenizer.as_os_str().is_empty() {
            return Err(CliError::MissingValue(
                "missing tokenizer path; pass it positionally or with `--tokenizer`",
            ));
        }

        Ok(cli)
    }
}

#[derive(Debug)]
enum CliError {
    MissingValue(&'static str),
    ParseValue {
        flag: &'static str,
        value: String,
    },
    Mode(generation::mode::ParseGenerationModeError),
    UnknownFlag(String),
    UnexpectedArgument(String),
    HelpRequested,
}

#[derive(Debug)]
enum AppError {
    Cli(CliError),
    Config(config::ConfigError),
    Weights(weights::WeightsError),
    Tokenizer(tokenizer::TokenizerError),
    Sampler(sampler::SamplerError),
    Transformer(transformer::TransformerError),
    Generate(generation::generate::GenerateError),
    Chat(generation::chat::ChatError),
}

fn next_value(
    args: &mut impl Iterator<Item = String>,
    flag: &'static str,
) -> Result<String, CliError> {
    args.next().ok_or(CliError::MissingValue(match flag {
        "--model" => "missing value for `--model`",
        "--tokenizer" => "missing value for `--tokenizer`",
        "--input" => "missing value for `--input`",
        "--mode" => "missing value for `--mode`",
        "--steps" => "missing value for `--steps`",
        "--temperature" => "missing value for `--temperature`",
        "--top-p" => "missing value for `--top-p`",
        "--seed" => "missing value for `--seed`",
        "--system" => "missing value for `--system`",
        _ => "missing value",
    }))
}

fn parse_value<T: FromStr>(value: &str, flag: &'static str) -> Result<T, CliError> {
    value.parse::<T>().map_err(|_| CliError::ParseValue {
        flag,
        value: value.to_string(),
    })
}

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingValue(message) => write!(f, "{message}"),
            Self::ParseValue { flag, value } => {
                write!(f, "invalid value `{value}` for {flag}")
            }
            Self::Mode(err) => write!(f, "{err}"),
            Self::UnknownFlag(flag) => write!(f, "unknown flag `{flag}`"),
            Self::UnexpectedArgument(arg) => write!(f, "unexpected argument `{arg}`"),
            Self::HelpRequested => write!(f, "{}", usage()),
        }
    }
}

impl std::error::Error for CliError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Mode(err) => Some(err),
            Self::MissingValue(_)
            | Self::ParseValue { .. }
            | Self::UnknownFlag(_)
            | Self::UnexpectedArgument(_)
            | Self::HelpRequested => None,
        }
    }
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cli(CliError::HelpRequested) => write!(f, "{}", usage()),
            Self::Cli(err) => write!(f, "{err}\n\n{}", usage()),
            Self::Config(err) => write!(f, "{err}"),
            Self::Weights(err) => write!(f, "{err}"),
            Self::Tokenizer(err) => write!(f, "{err}"),
            Self::Sampler(err) => write!(f, "{err}"),
            Self::Transformer(err) => write!(f, "{err}"),
            Self::Generate(err) => write!(f, "{err}"),
            Self::Chat(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Cli(err) => Some(err),
            Self::Config(err) => Some(err),
            Self::Weights(err) => Some(err),
            Self::Tokenizer(err) => Some(err),
            Self::Sampler(err) => Some(err),
            Self::Transformer(err) => Some(err),
            Self::Generate(err) => Some(err),
            Self::Chat(err) => Some(err),
        }
    }
}

fn usage() -> &'static str {
    "Usage:
  llama-rs --model MODEL --tokenizer TOKENIZER [options]
  llama-rs MODEL TOKENIZER [INPUT] [options]

Options:
  -m, --mode MODE           `generate` (default) or `chat`
  -i, --input TEXT          prompt text
  --model PATH              path to model.bin
  -t, --tokenizer PATH      path to tokenizer.bin
  --steps N                 max generated tokens (default: 128)
  --temp, --temperature T   sampling temperature (default: 1.0)
  --top-p P                 nucleus sampling threshold (default: 0.9)
  --seed N                  random seed (default: 42)
  --system TEXT             system prompt for chat mode
  -h, --help                show this message"
}

#[cfg(test)]
mod tests {
    use super::{Cli, CliError, GenerationMode};

    #[test]
    fn cli_defaults_to_generate_mode() {
        let cli = Cli::parse([
            "model.bin".to_string(),
            "tokenizer.bin".to_string(),
            "-i".to_string(),
            "hello".to_string(),
        ])
        .expect("cli should parse");

        assert_eq!(cli.mode, GenerationMode::Generate);
        assert_eq!(cli.input.as_deref(), Some("hello"));
    }

    #[test]
    fn cli_parses_chat_mode_from_m_flag() {
        let cli = Cli::parse([
            "--model".to_string(),
            "model.bin".to_string(),
            "--tokenizer".to_string(),
            "tokenizer.bin".to_string(),
            "-m".to_string(),
            "chat".to_string(),
            "-i".to_string(),
            "hello".to_string(),
        ])
        .expect("cli should parse");

        assert_eq!(cli.mode, GenerationMode::Chat);
    }

    #[test]
    fn cli_rejects_unknown_flags() {
        let err = Cli::parse([
            "model.bin".to_string(),
            "tokenizer.bin".to_string(),
            "--bad".to_string(),
        ])
        .expect_err("cli should fail");

        assert!(matches!(err, CliError::UnknownFlag(_)));
    }
}
