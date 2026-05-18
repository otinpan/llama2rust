use crate::transformer::Transformer;
use crate::tokenizer::Tokenizer;
use crate::sampler::Sampler;

// @trace-pilot e5b95bd8272094d29aa04e05b6163f3fa16d7413
// void chat
pub fn chat(
    transformer: &mut Transformer,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    user_prompt: Option<&str>,
    system_prompt: Option<&str>,
    steps: usize,
) -> std::io::Result<()>{
    todo!()
}

