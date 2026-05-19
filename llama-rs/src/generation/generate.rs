use crate::sampler::Sampler;
use crate::timer::Timer;
use crate::tokenizer::Tokenizer;
use crate::transformer::Transformer;

// @trace-pilot 26d3b064cd0a0f29fa2f9396bd83121745d3a33c
// void generate
pub fn generate(
    transformer: &mut Transformer,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    prompt: Option<&str>,
    steps: usize,
) -> std::io::Result<()> {
    let prompt = prompt.unwrap_or("");
    let bos = true;
    let eos = false;
    let prompt_tokens = tokenizer.encode(Some(prompt), bos, eos)?;

    if prompt_tokens.is_empty() {
        return Ok(());
    }

    let mut timer: Option<Timer> = None;
    let mut token = prompt_tokens[0];
    let num_prompt_tokens = prompt_tokens.len();
    let mut pos = 0usize;

    while pos < steps {
        let logits: Vec<f32> = transformer.forward(token, pos);

        let next = if pos < num_prompt_tokens.saturating_sub(1) {
            prompt_tokens[pos + 1]
        } else {
            sampler.sample(logits)
        };
        pos += 1;

        if next == 1 {
            break;
        }

        let piece = tokenizer.decode(token, next);
        print!("{}", piece);
        token = next;

        if timer.is_none() {
            timer = Some(Timer::start("generate"));
        }
    }

    println!();

    if pos > 1 {
        if let Some(timer) = &timer {
            let elapsed_ms = timer.elapsed_ms();
            if elapsed_ms > 0 {
                let tok_per_sec = ((pos - 1) as f64) * 1000.0 / elapsed_ms as f64;
                crate::log_info!("achieved tok/s: {}", tok_per_sec);
            }
            timer.finish();
        }
    }

    Ok(())
}
