# llama2rust

`llama2rust` is a small Rust implementation of a LLaMA-style transformer for learning and experimentation. It loads a binary checkpoint, tokenizes an input prompt, runs inference, and generates text from the command line. This project is originate [llama2.c](https://github.com/karpathy/llama2.c)

The repository also contains a C version for comparison, but the main Rust implementation lives in [`llama-rs`](./llama-rs).

## What It Does

- Loads a model checkpoint from a `.bin` file
- Loads a tokenizer from `tokenizer.bin`
- Supports `generate` mode for prompt completion
- Supports `chat` mode for multi-turn style interaction
- Exposes simple sampling controls such as temperature and top-p

## Requirements

- Rust toolchain with `cargo`
- A model checkpoint file such as `stories15M.bin`
- A tokenizer file at `llama-rs/tokenizer.bin`

The helper script in this repository expects:

- `stories15M.bin` at the repository root
- `llama-rs/tokenizer.bin`

## Build

```bash
cargo build --manifest-path llama-rs/Cargo.toml
```

The compiled binary will be created at:

```bash
llama-rs/target/debug/llama-rs
```

## Basic Usage

You can run the Rust version with the helper script:

```bash
./run_llama_rs.sh "Once upon a time"
```

This script uses:

- model: `stories15M.bin`
- tokenizer: `llama-rs/tokenizer.bin`
- mode: `generate`
- steps: `256`
- temperature: `0`

## Direct CLI Usage

You can also run the binary directly:

```bash
./llama-rs/target/debug/llama-rs stories15M.bin -m generate -i "Once upon a time" -z llama-rs/tokenizer.bin -n 256 -t 1.0 -p 0.9
```

### Options

- `-m <mode>`: `generate` or `chat`
- `-i <prompt>`: input prompt
- `-z <path>`: tokenizer path
- `-n <steps>`: number of generation steps
- `-t <temperature>`: sampling temperature
- `-p <top-p>`: top-p sampling value
- `-s <seed>`: random seed
- `-y <prompt>`: optional system prompt for chat mode

## Chat Example

```bash
./llama-rs/target/debug/llama-rs stories15M.bin -m chat -i "Hello" -y "You are a helpful assistant." -z llama-rs/tokenizer.bin -n 256
```

## Project Structure

- `llama-rs/src/main.rs`: CLI entry point
- `llama-rs/src/transformer.rs`: transformer inference
- `llama-rs/src/tokenizer.rs`: tokenizer loading and encoding/decoding
- `llama-rs/src/sampler.rs`: token sampling
- `llama-rs/src/generation/`: text generation and chat logic

## Notes

This project is intended to be simple and readable rather than production-oriented. It is best suited for understanding how checkpoint loading, tokenization, sampling, and transformer inference fit together in a minimal Rust codebase.
