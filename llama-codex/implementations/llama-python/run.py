# @trace-pilot 019e5092a1852351bdba2d2bb6b6716dc2c1e0a0
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from config import Config
from tokenizer import Tokenizer
from transformer import Transformer
from weights import load_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal llama runner in Python")
    parser.add_argument("model", type=Path, help="Path to model.bin")
    parser.add_argument("tokenizer", type=Path, help="Path to tokenizer.bin")
    parser.add_argument("-i", "--prompt", default="", help="Prompt text")
    parser.add_argument(
        "-n",
        "--max-new-tokens",
        type=int,
        default=32,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for greedy decoding",
    )
    parser.add_argument(
        "--bos",
        action="store_true",
        help="Prepend BOS token when encoding the prompt",
    )
    parser.add_argument(
        "--eos",
        action="store_true",
        help="Append EOS token to the prompt before generation",
    )
    parser.add_argument(
        "--dummy-prefix",
        action="store_true",
        help="Prepend a single space before tokenization",
    )
    return parser.parse_args()


def sample_next_token(
    logits: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if temperature <= 0.0:
        return int(np.argmax(logits))

    scaled = logits.astype(np.float64) / temperature
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= np.sum(probs)
    return int(rng.choice(len(probs), p=probs))


def generate(
    model_path: Path,
    tokenizer_path: Path,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    add_bos: bool,
    add_eos: bool,
    dummy_prefix: bool,
) -> str:
    config = Config.from_file(model_path)
    weights = load_weights(model_path, config)
    tokenizer = Tokenizer.from_file(tokenizer_path, config.vocab_size)
    transformer = Transformer(config, weights)

    prompt_tokens = tokenizer.encode(
        prompt,
        add_bos=add_bos,
        add_eos=add_eos,
        dummy_prefix=dummy_prefix,
    )
    if not prompt_tokens:
        prompt_tokens = [tokenizer.bos_id]

    if len(prompt_tokens) >= config.seq_len:
        raise ValueError(
            f"prompt is too long: {len(prompt_tokens)} tokens for seq_len={config.seq_len}"
        )

    rng = np.random.default_rng()
    logits: np.ndarray | None = None
    pos = 0

    for pos, token in enumerate(prompt_tokens):
        logits = transformer.forward(token, pos)

    generated: list[int] = []
    current_pos = pos
    current_token = prompt_tokens[-1]

    for _ in range(max_new_tokens):
        if logits is None:
            logits = transformer.forward(current_token, current_pos)

        next_token = sample_next_token(logits, temperature, rng)
        if next_token == tokenizer.eos_id:
            break

        generated.append(next_token)
        current_pos += 1
        if current_pos >= config.seq_len:
            break
        current_token = next_token
        logits = transformer.forward(current_token, current_pos)

    return tokenizer.decode(generated, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    text = generate(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        add_bos=args.bos,
        add_eos=args.eos,
        dummy_prefix=args.dummy_prefix,
    )
    print(text, end="")


if __name__ == "__main__":
    main()
