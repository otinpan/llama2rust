use crate::transformer::Transformer;
use crate::tokenizer::Tokenizer;
use crate::sampler::Sampler;

// @trace-pilot 26d3b064cd0a0f29fa2f9396bd83121745d3a33c
// void generate
pub fn generate(
    transformer: &mut Transformer,
    tokenizer: &mut Tokenizer,
    sampler: &mut Sampler,
    prompt: Option<&str>,
    steps: usize,
) -> std::io::Result<()>{
    let prompt=prompt.unwrap_or("");

    let tokens=tokenizer.encode(Some(prompt),true,false);

    // encoder
    todo!()
}


