mod config;
mod checkpoint;
mod kernels;
mod sampler;
mod state;
mod tokenizer;
mod transformer;
mod weights;

use std::{ffi::OsString, path::PathBuf};

use checkpoint::load_checkpoint;
use config::Config;
use sampler::Sampler;
use tokenizer::Tokenizer;
use transformer::Transformer;
use weights::TransformerWeights;

fn main() {
    let cli = parse_cli(std::env::args_os().skip(1).collect());

    if let Some(path) = cli.checkpoint_path {
        let token = cli.token.unwrap_or(1);
        let pos = cli.pos.unwrap_or(0);
        let steps = cli.steps.unwrap_or(1);
        let temperature = cli.temperature.unwrap_or(0.0);
        let topp = cli.topp.unwrap_or(0.9);
        let seed = cli.seed.unwrap_or(1);

        match load_checkpoint(&path) {
            Ok(checkpoint) => {
                let mut transformer = Transformer::new(checkpoint.config, checkpoint.weights);
                let config = transformer.config().clone();
                let mut sampler = Sampler::new(config.vocab_size, temperature, topp, seed);

                println!(
                    "checkpoint loaded: dim={}, layers={}, vocab={}",
                    config.dim, config.n_layers, config.vocab_size
                );

                if let Some(tokenizer_path) = cli.tokenizer_path {
                    let prompt = cli.prompt.as_deref().unwrap_or("");
                    validate_generation_args(pos, steps, &config);
                    let tokenizer = Tokenizer::from_file(&tokenizer_path, config.vocab_size)
                        .unwrap_or_else(|error| {
                            eprintln!(
                                "failed to load tokenizer {}: {error}",
                                tokenizer_path.display()
                            );
                            std::process::exit(1);
                        });
                    let generated = generate_prompt(
                        &mut transformer,
                        &tokenizer,
                        &mut sampler,
                        prompt,
                        pos,
                        steps,
                    );
                    println!(
                        "generate ok: prompt={:?}, start_pos={}, steps={}, tokens={:?}",
                        prompt, pos, steps, generated.tokens
                    );
                    if !generated.text.is_empty() {
                        println!("decoded: {}", generated.text);
                    }
                } else {
                    validate_inference_args(token, pos, &config);
                    if steps <= 1 {
                        let logits = transformer.forward(token, pos);
                        let (argmax_token, argmax_logit) = argmax(logits);
                        println!(
                            "forward ok: token={}, pos={}, logits={}, argmax_token={}, argmax_logit={:.6}",
                            token,
                            pos,
                            logits.len(),
                            argmax_token,
                            argmax_logit
                        );
                    } else {
                        validate_generation_args(pos, steps, &config);
                        let generated =
                            generate_tokens(&mut transformer, &mut sampler, token, pos, steps);
                        println!(
                            "generate ok: start_token={}, start_pos={}, steps={}, tokens={:?}",
                            token, pos, steps, generated
                        );
                    }
                }
            }
            Err(error) => {
                eprintln!("failed to load checkpoint {}: {error}", path.display());
                std::process::exit(1);
            }
        }
    } else {
        let config = Config::new(288, 768, 6, 6, 6, 32_000, 256);
        let weights = TransformerWeights::new(&config);
        let transformer = Transformer::new(config, weights);

        println!(
            "transformer skeleton ready: dim={}, layers={}, vocab={}",
            transformer.config().dim,
            transformer.config().n_layers,
            transformer.config().vocab_size
        );
    }
}

#[derive(Default)]
struct Cli {
    checkpoint_path: Option<PathBuf>,
    tokenizer_path: Option<PathBuf>,
    prompt: Option<String>,
    token: Option<usize>,
    pos: Option<usize>,
    steps: Option<usize>,
    temperature: Option<f32>,
    topp: Option<f32>,
    seed: Option<u64>,
}

struct GeneratedText {
    tokens: Vec<usize>,
    text: String,
}

fn parse_cli(args: Vec<OsString>) -> Cli {
    let mut cli = Cli::default();
    let mut positionals = Vec::new();
    let mut index = 0;

    while index < args.len() {
        let value = os_to_string(args[index].clone(), "argument");
        match value.as_str() {
            "--tokenizer" => {
                index += 1;
                cli.tokenizer_path = Some(PathBuf::from(require_arg(&args, index, "--tokenizer")));
            }
            "--prompt" => {
                index += 1;
                cli.prompt = Some(require_arg(&args, index, "--prompt"));
            }
            "--temperature" => {
                index += 1;
                cli.temperature = Some(parse_f32(&require_arg(&args, index, "--temperature"), "temperature"));
            }
            "--topp" => {
                index += 1;
                cli.topp = Some(parse_f32(&require_arg(&args, index, "--topp"), "topp"));
            }
            "--seed" => {
                index += 1;
                cli.seed = Some(parse_u64(&require_arg(&args, index, "--seed"), "seed"));
            }
            _ => positionals.push(value),
        }
        index += 1;
    }

    cli.checkpoint_path = positionals.first().map(PathBuf::from);
    cli.token = positionals.get(1).map(|value| parse_usize(value, "token"));
    cli.pos = positionals.get(2).map(|value| parse_usize(value, "pos"));
    cli.steps = positionals.get(3).map(|value| parse_usize(value, "steps"));
    cli
}

fn require_arg(args: &[OsString], index: usize, name: &str) -> String {
    args.get(index)
        .cloned()
        .map(|arg| os_to_string(arg, name))
        .unwrap_or_else(|| {
            eprintln!("missing value for {name}");
            std::process::exit(1);
        })
}

fn os_to_string(arg: OsString, name: &str) -> String {
    arg.into_string().unwrap_or_else(|_| {
        eprintln!("{name} must be valid utf-8");
        std::process::exit(1);
    })
}

fn parse_usize(value: &str, name: &str) -> usize {
    value.parse::<usize>().unwrap_or_else(|error| {
        eprintln!("invalid {name} '{value}': {error}");
        std::process::exit(1);
    })
}

fn parse_u64(value: &str, name: &str) -> u64 {
    value.parse::<u64>().unwrap_or_else(|error| {
        eprintln!("invalid {name} '{value}': {error}");
        std::process::exit(1);
    })
}

fn parse_f32(value: &str, name: &str) -> f32 {
    value.parse::<f32>().unwrap_or_else(|error| {
        eprintln!("invalid {name} '{value}': {error}");
        std::process::exit(1);
    })
}

fn validate_inference_args(token: usize, pos: usize, config: &Config) {
    if token >= config.vocab_size {
        eprintln!(
            "token out of range: token={} vocab_size={}",
            token, config.vocab_size
        );
        std::process::exit(1);
    }
    if pos >= config.seq_len {
        eprintln!("pos out of range: pos={} seq_len={}", pos, config.seq_len);
        std::process::exit(1);
    }
}

fn validate_generation_args(pos: usize, steps: usize, config: &Config) {
    if pos + steps > config.seq_len {
        eprintln!(
            "generation exceeds seq_len: start_pos={} steps={} seq_len={}",
            pos, steps, config.seq_len
        );
        std::process::exit(1);
    }
}

fn generate_tokens(
    transformer: &mut Transformer,
    sampler: &mut Sampler,
    start_token: usize,
    start_pos: usize,
    steps: usize,
) -> Vec<usize> {
    transformer.reset_state();

    let mut token = start_token;
    let mut generated = Vec::with_capacity(steps);

    for step in 0..steps {
        let pos = start_pos + step;
        let logits = transformer.forward(token, pos);
        let next_token = sampler.sample(logits);
        generated.push(next_token);
        token = next_token;
    }

    generated
}

fn generate_prompt(
    transformer: &mut Transformer,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt: &str,
    start_pos: usize,
    steps: usize,
) -> GeneratedText {
    transformer.reset_state();

    let prompt_tokens = tokenizer.encode(prompt, true, false).unwrap_or_else(|error| {
        eprintln!("failed to encode prompt: {error}");
        std::process::exit(1);
    });
    if prompt_tokens.is_empty() {
        eprintln!("prompt encoding produced no tokens");
        std::process::exit(1);
    }

    let mut token = prompt_tokens[0];
    let mut generated_tokens = Vec::with_capacity(steps);
    let mut decoded = String::new();

    for step in 0..steps {
        let pos = start_pos + step;
        let logits = transformer.forward(token, pos);
        let next = if step < prompt_tokens.len() - 1 {
            prompt_tokens[step + 1]
        } else {
            sampler.sample(logits)
        };

        generated_tokens.push(next);
        if let Some(piece) = tokenizer.safe_decode(token, next).unwrap_or_else(|error| {
            eprintln!("failed to decode token {next}: {error}");
            std::process::exit(1);
        }) {
            decoded.push_str(&piece);
        }
        token = next;
    }

    GeneratedText {
        tokens: generated_tokens,
        text: decoded,
    }
}

fn argmax(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .copied()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .unwrap_or_else(|| {
            eprintln!("logits are empty");
            std::process::exit(1);
        })
}

#[cfg(test)]
mod tests {
    use crate::{
        config::Config,
        sampler::Sampler,
        tokenizer::Tokenizer,
        transformer::Transformer,
        weights::TransformerWeights,
    };

    use super::{generate_prompt, generate_tokens};
    use std::io::Cursor;

    #[test]
    fn generates_multiple_tokens_with_sampler_loop() {
        let config = Config::new(4, 6, 1, 2, 2, 8, 4);
        let mut weights = TransformerWeights::new(&config);
        weights.token_embedding_table[4..8].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        weights.rms_final_weight.fill(1.0);
        weights.wcls.copy_from_slice(&rectangular_identity(8, 4));

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
        let mut sampler = Sampler::new(8, 0.0, 0.9, 1);
        let generated = generate_tokens(&mut transformer, &mut sampler, 1, 0, 3);

        assert_eq!(generated.len(), 3);
        assert!(generated.iter().all(|token| *token < 8));
    }

    #[test]
    fn generates_from_prompt_tokens() {
        let config = Config::new(4, 6, 1, 2, 2, 8, 8);
        let mut weights = TransformerWeights::new(&config);
        weights.token_embedding_table[4..8].copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        weights.rms_final_weight.fill(1.0);
        weights.wcls.copy_from_slice(&rectangular_identity(8, 4));

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

        let mut cursor = Cursor::new(tokenizer_fixture_bytes());
        let tokenizer = Tokenizer::from_reader(&mut cursor, 8).unwrap();

        let mut transformer = Transformer::new(config, weights);
        let mut sampler = Sampler::new(8, 0.0, 0.9, 1);
        let generated = generate_prompt(&mut transformer, &tokenizer, &mut sampler, "ab", 0, 4);

        assert_eq!(generated.tokens.len(), 4);
        assert!(!generated.text.is_empty());
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

    fn tokenizer_fixture_bytes() -> Vec<u8> {
        let pieces = [
            ("<unk>", 0.0_f32),
            ("<s>", 0.0),
            ("</s>", 0.0),
            (" ", 0.0),
            ("a", 1.0),
            ("b", 1.0),
            ("<0x61>", 0.0),
            ("ab", 10.0),
        ];

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(8_u32).to_le_bytes());
        for (piece, score) in pieces {
            bytes.extend_from_slice(&score.to_le_bytes());
            bytes.extend_from_slice(&(piece.len() as u32).to_le_bytes());
            bytes.extend_from_slice(piece.as_bytes());
        }
        bytes
    }
}
