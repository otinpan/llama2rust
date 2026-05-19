use llama_rs::transformer::Transformer;
use llama_rs::tokenizer::Tokenizer;
use llama_rs::generation::generate::generate;
use llama_rs::generation::chat::chat;
// @trace-pilot 5f50038dae75a7ab6c556f586a9adb5d86c3b026
use llama_rs::logger;

// @trace-pilot 3624d0606abc5bcc94a91c7153eced4e02d4465c
#[derive(Debug)]
struct CliOptions {
    checkpoint: String,
    tokenizer_path: String,
    temperature: f32,
    topp: f32,
    rng_seed: u64,
    steps: usize,
    prompt: Option<String>,
    mode: String,
    system_prompt: Option<String>,
}

// @trace-pilot 3624d0606abc5bcc94a91c7153eced4e02d4465c
fn usage(program: &str) -> String {
    format!(
        "Usage:   {program} <checkpoint> [options]\n\
         Example: {program} model.bin -n 256 -i \"Once upon a time\"\n\
         Options:\n\
           -t <float>  temperature in [0, inf), default 1.0\n\
           -p <float>  top-p sampling value in [0, 1], default 0.9\n\
           -s <int>    random seed, default 0\n\
           -n <int>    number of steps to run for, default 256; 0 = max_seq_len\n\
           -i <string> input prompt\n\
           -z <string> optional path to custom tokenizer\n\
           -m <string> mode: generate|chat, default generate\n\
           -y <string> optional system prompt in chat mode"
    )
}

// @trace-pilot 3624d0606abc5bcc94a91c7153eced4e02d4465c
fn parse_args() -> Result<CliOptions, String> {
    let mut args = std::env::args();
    let program = args.next().unwrap_or_else(|| "llama-rs".to_string());
    let checkpoint = args.next().ok_or_else(|| usage(&program))?;

    let mut options = CliOptions {
        checkpoint,
        tokenizer_path: "tokenizer.bin".to_string(),
        temperature: 1.0,
        topp: 0.9,
        rng_seed: 0,
        steps: 256,
        prompt: None,
        mode: "generate".to_string(),
        system_prompt: None,
    };

    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| usage(&program))?;
        match flag.as_str() {
            "-t" => {
                options.temperature = value.parse().map_err(|_| {
                    format!("invalid value for -t: {value}\n\n{}", usage(&program))
                })?;
            }
            "-p" => {
                options.topp = value.parse().map_err(|_| {
                    format!("invalid value for -p: {value}\n\n{}", usage(&program))
                })?;
            }
            "-s" => {
                options.rng_seed = value.parse().map_err(|_| {
                    format!("invalid value for -s: {value}\n\n{}", usage(&program))
                })?;
            }
            "-n" => {
                options.steps = value.parse().map_err(|_| {
                    format!("invalid value for -n: {value}\n\n{}", usage(&program))
                })?;
            }
            "-i" => {
                options.prompt = Some(value);
            }
            "-z" => {
                options.tokenizer_path = value;
            }
            "-m" => {
                options.mode = value;
            }
            "-y" => {
                options.system_prompt = Some(value);
            }
            _ => {
                return Err(format!("unknown option: {flag}\n\n{}", usage(&program)));
            }
        }
    }

// @trace-pilot 3624d0606abc5bcc94a91c7153eced4e02d4465c
    Ok(options)
}

// @trace-pilot 3624d0606abc5bcc94a91c7153eced4e02d4465c
fn main() {
// @trace-pilot 5f50038dae75a7ab6c556f586a9adb5d86c3b026
    logger::init();

    let mut options = parse_args().unwrap_or_else(|message| {
        eprintln!("{message}");
        std::process::exit(1);
    });

    let transformer = Transformer::new(&options.checkpoint).expect("failed to open checkpoint");
    options.steps = transformer.clamp_steps(options.steps);


    let tokanizer=Tokenizer::new(
        &options.tokenizer_path,
        transformer.config().vocab_size
    ).unwrap();

    println!("config: {:?}", transformer.config());
    println!("options: {:?}", options);
}
