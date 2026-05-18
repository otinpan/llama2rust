use crate::transformer::Transformer;
use crate::tokenizer::Tokenizer;
use crate::sampler::Sampler;

// @trace-pilot 26d3b064cd0a0f29fa2f9396bd83121745d3a33c
// void generate
pub fn generate(
    transformer: &mut Transformer,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    prompt_in: Option<&str>,
    steps: usize,
) -> std::io::Result<()>{
    todo!()
}


