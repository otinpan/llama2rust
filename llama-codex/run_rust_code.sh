#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT_DIR/implementations-from-code/llama-rs/target/debug/llama-rs"
PROMPT="${*:-}"

if [[ -z "$PROMPT" ]]; then
  echo "usage: $0 <prompt>" >&2
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "binary not found: $BIN" >&2
  echo "build it first with: cargo build --manifest-path $ROOT_DIR/implementations-from-code/llama-rs/Cargo.toml" >&2
  exit 1
fi

"$BIN" \
  "$ROOT_DIR/model.bin" \
  0 \
  0 \
  256 \
  --tokenizer "$ROOT_DIR/tokenizer.bin" \
  --prompt "$PROMPT" \
  --temperature 0
