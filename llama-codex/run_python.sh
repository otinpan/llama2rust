#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT="${*:-}"

if [[ -z "$PROMPT" ]]; then
  echo "usage: $0 <prompt>" >&2
  exit 1
fi

"$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/implementations/python-llama/run.py" \
  "$ROOT_DIR/model.bin" \
  "$ROOT_DIR/tokenizer.bin" \
  -i "$PROMPT" \
  -n 32 \
  --temperature 0
