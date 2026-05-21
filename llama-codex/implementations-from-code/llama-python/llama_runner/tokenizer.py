from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import bisect
import re
import struct


BYTE_TOKEN_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


@dataclass(slots=True)
class Tokenizer:
    vocab: list[str]
    vocab_scores: list[float]
    max_token_length: int
    sorted_vocab: list[tuple[str, int]] = field(default_factory=list)
    byte_pieces: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path, vocab_size: int) -> "Tokenizer":
        tokenizer_path = Path(path)
        with tokenizer_path.open("rb") as handle:
            max_token_length = struct.unpack("i", handle.read(4))[0]
            vocab: list[str] = []
            vocab_scores: list[float] = []
            for _ in range(vocab_size):
                vocab_scores.append(struct.unpack("f", handle.read(4))[0])
                token_len = struct.unpack("i", handle.read(4))[0]
                token_bytes = handle.read(token_len)
                vocab.append(token_bytes.decode("utf-8", errors="strict"))
        tokenizer = cls(
            vocab=vocab,
            vocab_scores=vocab_scores,
            max_token_length=max_token_length,
        )
        tokenizer._build_index()
        return tokenizer

    def _build_index(self) -> None:
        self.sorted_vocab = sorted((token, idx) for idx, token in enumerate(self.vocab))
        self.byte_pieces = [bytes([i]).decode("latin-1") for i in range(256)]

    def str_lookup(self, text: str) -> int:
        idx = bisect.bisect_left(self.sorted_vocab, (text, -1))
        if idx < len(self.sorted_vocab) and self.sorted_vocab[idx][0] == text:
            return self.sorted_vocab[idx][1]
        return -1

    def decode_token(self, prev_token: int, token: int) -> str:
        piece = self.vocab[token]
        if prev_token == 1 and piece.startswith(" "):
            piece = piece[1:]
        match = BYTE_TOKEN_RE.match(piece)
        if match:
            byte_value = int(match.group(1), 16)
            piece = self.byte_pieces[byte_value]
        return piece

    def safe_piece(self, piece: str) -> str:
        if not piece:
            return ""
        if len(piece) == 1:
            ch = piece[0]
            if not (ch.isprintable() or ch.isspace()):
                return ""
        return piece

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        if text is None:
            raise ValueError("cannot encode None text")

        tokens: list[int] = []
        if bos:
            tokens.append(1)

        if text:
            dummy_prefix = self.str_lookup(" ")
            if dummy_prefix == -1:
                raise ValueError("dummy prefix token not found in vocabulary")
            tokens.append(dummy_prefix)

        raw_bytes = text.encode("utf-8")
        buffer = bytearray()
        for i, value in enumerate(raw_bytes):
            if value & 0xC0 != 0x80:
                buffer.clear()
            buffer.append(value)
            if i + 1 < len(raw_bytes) and raw_bytes[i + 1] & 0xC0 == 0x80 and len(buffer) < 4:
                continue

            piece_bytes = bytes(buffer)
            try:
                piece_text = piece_bytes.decode("utf-8")
            except UnicodeDecodeError:
                piece_text = ""

            token_id = self.str_lookup(piece_text) if piece_text else -1
            if token_id != -1:
                tokens.append(token_id)
            else:
                for byte in piece_bytes:
                    tokens.append(byte + 3)
            buffer.clear()

        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1
            for i in range(len(tokens) - 1):
                merged = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                token_id = self.str_lookup(merged)
                if token_id != -1 and self.vocab_scores[token_id] > best_score:
                    best_score = self.vocab_scores[token_id]
                    best_id = token_id
                    best_idx = i
            if best_idx == -1:
                break
            tokens[best_idx] = best_id
            del tokens[best_idx + 1]

        if eos:
            tokens.append(2)
        return tokens
