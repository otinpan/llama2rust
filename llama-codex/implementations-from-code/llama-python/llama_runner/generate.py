from __future__ import annotations

import time

from .model import Transformer
from .sampler import Sampler
from .tokenizer import Tokenizer


def generate(
    transformer: Transformer,
    tokenizer: Tokenizer,
    sampler: Sampler,
    prompt: str | None,
    steps: int,
) -> tuple[str, float | None]:
    if prompt is None:
        prompt = ""

    prompt_tokens = tokenizer.encode(prompt, bos=True, eos=False)
    if not prompt_tokens:
        raise ValueError("expected at least one prompt token")

    pieces: list[str] = []
    start_time: float | None = None
    token = prompt_tokens[0]
    pos = 0

    while pos < steps:
        logits = transformer.forward(token, pos)
        if pos < len(prompt_tokens) - 1:
            next_token = prompt_tokens[pos + 1]
        else:
            next_token = sampler.sample(logits)
        pos += 1

        if next_token == 1:
            break

        piece = tokenizer.safe_piece(tokenizer.decode_token(token, next_token))
        if piece:
            pieces.append(piece)
        token = next_token

        if start_time is None:
            start_time = time.time()

    tok_per_sec: float | None = None
    if start_time is not None and pos > 1:
        elapsed = time.time() - start_time
        if elapsed > 0:
            tok_per_sec = (pos - 1) / elapsed
    return "".join(pieces), tok_per_sec
