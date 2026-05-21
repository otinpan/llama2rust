from __future__ import annotations

from dataclasses import dataclass
import time

from .model import Transformer
from .sampler import Sampler
from .tokenizer import Tokenizer


@dataclass(slots=True)
class ChatTurn:
    user: str
    assistant: str


def render_prompt(user_prompt: str, system_prompt: str | None, pos: int) -> str:
    if pos == 0 and system_prompt:
        return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
    return f"[INST] {user_prompt} [/INST]"


def complete_assistant_turn(
    transformer: Transformer,
    tokenizer: Tokenizer,
    sampler: Sampler,
    prompt_tokens: list[int],
    pos: int,
    max_steps: int,
) -> tuple[str, int, float | None]:
    user_idx = 0
    next_token = 0
    pieces: list[str] = []
    start_time: float | None = None

    while pos < max_steps:
        if user_idx < len(prompt_tokens):
            token = prompt_tokens[user_idx]
            user_idx += 1
        else:
            token = next_token

        if token == 2:
            break

        logits = transformer.forward(token, pos)
        next_token = sampler.sample(logits)
        pos += 1

        if user_idx >= len(prompt_tokens) and next_token != 2:
            piece = tokenizer.safe_piece(tokenizer.decode_token(token, next_token))
            if piece:
                pieces.append(piece)

        if next_token == 2:
            break

        if start_time is None:
            start_time = time.time()

    tok_per_sec: float | None = None
    if start_time is not None and pos > 1:
        elapsed = time.time() - start_time
        if elapsed > 0:
            tok_per_sec = (pos - 1) / elapsed
    return "".join(pieces), pos, tok_per_sec


def chat_once(
    transformer: Transformer,
    tokenizer: Tokenizer,
    sampler: Sampler,
    user_prompt: str,
    system_prompt: str | None,
    steps: int,
) -> tuple[str, float | None]:
    transformer.reset_state()
    rendered_prompt = render_prompt(user_prompt, system_prompt, pos=0)
    prompt_tokens = tokenizer.encode(rendered_prompt, bos=True, eos=False)
    response, _, tok_per_sec = complete_assistant_turn(
        transformer=transformer,
        tokenizer=tokenizer,
        sampler=sampler,
        prompt_tokens=prompt_tokens,
        pos=0,
        max_steps=steps,
    )
    return response, tok_per_sec
