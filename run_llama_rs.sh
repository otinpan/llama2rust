#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT_DIR/llama-rs/target/debug/llama-rs"
TOKENIZER="$ROOT_DIR/llama-rs/tokenizer.bin"
MODEL="$ROOT_DIR/stories15M.bin"
PROMPT="${*:-}"

if [[ -z "$PROMPT" ]]; then
  echo "usage: $0 <prompt>" >&2
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN" >&2
  echo "build it first with: cargo build --manifest-path \"$ROOT_DIR/llama-rs/Cargo.toml\"" >&2
  exit 1
fi

if [[ ! -f "$TOKENIZER" ]]; then
  echo "tokenizer not found: $TOKENIZER" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "model not found: $MODEL" >&2
  exit 1
fi

"$BIN" \
  "$MODEL" \
  -m generate \
  -i "$PROMPT" \
  -z "$TOKENIZER" \
  -n 256 \
  -t 0
