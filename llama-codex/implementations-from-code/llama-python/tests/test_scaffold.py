from __future__ import annotations

import struct
from pathlib import Path

from llama_runner.config import Config
from llama_runner.tokenizer import Tokenizer


def write_tokenizer(path: Path, entries: list[tuple[float, bytes]]) -> None:
    with path.open("wb") as handle:
        handle.write(struct.pack("i", 8))
        for score, token_bytes in entries:
            handle.write(struct.pack("f", score))
            handle.write(struct.pack("i", len(token_bytes)))
            handle.write(token_bytes)


def test_config_header_roundtrip() -> None:
    config = Config(64, 256, 4, 8, 8, 32000, 128)
    data = Config.STRUCT.pack(
        config.dim,
        config.hidden_dim,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.vocab_size,
        config.seq_len,
    )
    restored = Config.from_bytes(data)
    assert restored == config


def test_tokenizer_encode_decode(tmp_path: Path) -> None:
    vocab = [
        (0.0, b"<unk>"),
        (0.0, b"<s>"),
        (0.0, b"</s>"),
        (0.1, b" "),
        (0.2, b"H"),
        (0.2, b"i"),
        (1.0, b"Hi"),
    ]
    tokenizer_path = tmp_path / "tokenizer.bin"
    write_tokenizer(tokenizer_path, vocab)
    tokenizer = Tokenizer.from_file(tokenizer_path, len(vocab))

    tokens = tokenizer.encode("Hi", bos=True, eos=False)
    assert tokens == [1, 3, 6]
    assert tokenizer.decode_token(1, 3) == ""
    assert tokenizer.decode_token(3, 6) == "Hi"
