#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$ROOT_DIR/implementations-from-code/llama-rs/Cargo.toml"
PROMPT="${*:-}"

if [[ -z "$PROMPT" ]]; then
  echo "usage: $0 <prompt>" >&2
  exit 1
fi

cargo run --quiet --manifest-path "$MANIFEST" -- \
  "$ROOT_DIR/model.bin" \
  0 \
  0 \
  256 \
  --tokenizer "$ROOT_DIR/tokenizer.bin" \
  --prompt "$PROMPT" \
  --temperature 0
