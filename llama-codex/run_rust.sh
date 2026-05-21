#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT_DIR/implementations/llama-rs/target/debug/llama-rs"
PROMPT="${*:-}"

if [[ -z "$PROMPT" ]]; then
  echo "usage: $0 <prompt>" >&2
  exit 1
fi

"$BIN" \
  --model "$ROOT_DIR/model.bin" \
  --tokenizer "$ROOT_DIR/tokenizer.bin" \
  -m generate \
  -i "$PROMPT" \
  --steps 256 \
  --temperature 0
