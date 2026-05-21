#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT_DIR/llama2.c"
BIN="$ROOT_DIR/run"
TOKENIZER="$ROOT_DIR/llama-rs/tokenizer.bin"
MODEL="$ROOT_DIR/stories15M.bin"
PROMPT="${*:-}"

if [[ -z "$PROMPT" ]]; then
  echo "usage: $0 <prompt>" >&2
  exit 1
fi

if [[ ! -f "$SRC" ]]; then
  echo "source not found: $SRC" >&2
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

if [[ ! -x "$BIN" || "$SRC" -nt "$BIN" ]]; then
  gcc -O3 -fopenmp -o "$BIN" "$SRC" -lm
fi

"$BIN" \
  "$MODEL" \
  -z "$TOKENIZER" \
  -m generate \
  -i "$PROMPT" \
  -n 256 \
  -t 0
