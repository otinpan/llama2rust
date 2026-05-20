use crate::generation::common::read_stdin;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;

use std::io::{self, Write};

// @trace-pilot e5b95bd8272094d29aa04e05b6163f3fa16d7413
// void chat
pub fn chat(
    transformer: &mut Transformer,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    cli_user_prompt: Option<&str>,
    cli_system_prompt: Option<&str>,
    steps: usize,
) -> io::Result<()> {
    let mut system_prompt = String::new();
    let mut prompt_tokens: Vec<u32> = Vec::new();
    let mut user_idx = 0usize;

    let mut user_turn = true;
    let mut next = 0u32;
    let mut token = 0u32;
    let mut pos = 0usize;

    while pos < steps {
        if user_turn {
            if pos == 0 {
                system_prompt = match cli_system_prompt {
                    Some(prompt) => prompt.to_string(),
                    None => read_stdin("Enter system prompt (optional): ")?,
                };
            }

            let user_prompt = if pos == 0 {
                match cli_user_prompt {
                    Some(prompt) => prompt.to_string(),
                    None => read_stdin("User: ")?,
                }
            } else {
                read_stdin("User: ")?
            };

            let rendered_prompt = if pos == 0 && !system_prompt.is_empty() {
                format!(
                    "[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
                    system_prompt, user_prompt
                )
            } else {
                format!("[INST] {} [/INST]", user_prompt)
            };

            prompt_tokens = tokenizer.encode(Some(&rendered_prompt), true, false)?;
            user_idx = 0;
            user_turn = false;
            print!("Assistant: ");
            io::stdout().flush()?;
        }

        if user_idx < prompt_tokens.len() {
            token = prompt_tokens[user_idx];
            user_idx += 1;
        } else {
            token = next;
        }

        if token == 2 {
            user_turn = true;
        }

        let logits = transformer.forward(token, pos);
        next = sampler.sample(logits);
        pos += 1;

        if user_idx >= prompt_tokens.len() && next != 2 {
            let piece = tokenizer.decode(token, next);
            if let Some(piece) = tokenizer.safe_piece(&piece) {
                print!("{piece}");
                io::stdout().flush()?;
            }
        }

        if next == 2 {
            println!();
            user_turn = true;
        }
    }

    println!();
    Ok(())
}
