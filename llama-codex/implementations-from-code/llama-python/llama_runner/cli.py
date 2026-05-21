from __future__ import annotations

import argparse

from .chat import chat_once
from .generate import generate
from .model import Transformer
from .sampler import Sampler
from .tokenizer import Tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Llama runner scaffold based on llama2.c")
    parser.add_argument("checkpoint", help="Path to model.bin")
    parser.add_argument("--mode", choices=("generate", "chat"), default="generate", help="Execution mode")
    parser.add_argument("--tokenizer", help="Path to tokenizer.bin")
    parser.add_argument("--prompt", default="", help="Prompt text for generation")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt for chat mode")
    parser.add_argument("--steps", type=int, default=32, help="Generation steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--topp", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--token", type=int, default=None, help="Input token id for single-step debug")
    parser.add_argument("--pos", type=int, default=0, help="Sequence position for single-step debug")
    parser.add_argument("--topk", type=int, default=10, help="How many logits to print")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    transformer = Transformer.from_file(args.checkpoint)
    try:
        print("config:", transformer.config)
        if args.token is not None:
            logits = transformer.forward(args.token, args.pos)
            top_indices = logits.argsort()[-args.topk :][::-1]
            print("top logits:")
            for token_id in top_indices:
                print(f"{int(token_id):6d} {float(logits[token_id]):.6f}")
            return

        if not args.tokenizer:
            raise SystemExit("--tokenizer is required unless --token is used")

        tokenizer = Tokenizer.from_file(args.tokenizer, transformer.config.vocab_size)
        sampler = Sampler(
            vocab_size=transformer.config.vocab_size,
            temperature=max(args.temperature, 0.0),
            topp=args.topp,
            rng_state=args.seed,
        )
        steps = min(args.steps, transformer.config.seq_len)
        if args.mode == "chat":
            text, tok_per_sec = chat_once(
                transformer=transformer,
                tokenizer=tokenizer,
                sampler=sampler,
                user_prompt=args.prompt,
                system_prompt=args.system_prompt,
                steps=steps,
            )
        else:
            text, tok_per_sec = generate(
                transformer=transformer,
                tokenizer=tokenizer,
                sampler=sampler,
                prompt=args.prompt,
                steps=steps,
            )
        print(text)
        if tok_per_sec is not None:
            print(f"tok/s: {tok_per_sec:.3f}")
    finally:
        transformer.close()


if __name__ == "__main__":
    main()
