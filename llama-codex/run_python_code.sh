#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT="${*:-}"
PY_APP_DIR="$ROOT_DIR/implementations-from-code/llama-python"

if [[ -z "$PROMPT" ]]; then
  echo "usage: $0 <prompt>" >&2
  exit 1
fi

PYTHONPATH="$PY_APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$ROOT_DIR/.venv/bin/python" -m llama_runner.cli \
  "$ROOT_DIR/model.bin" \
  --tokenizer "$ROOT_DIR/tokenizer.bin" \
  --prompt "$PROMPT" \
  --steps 256 \
  --temperature 0
