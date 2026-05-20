# @trace-pilot a145645e0bfc237ed8c280a5d540c6bce6d4f222
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


TOKENIZER_HEADER_FORMAT = "<I"
TOKENIZER_HEADER_SIZE = struct.calcsize(TOKENIZER_HEADER_FORMAT)
VOCAB_ENTRY_PREFIX_FORMAT = "<fi"
VOCAB_ENTRY_PREFIX_SIZE = struct.calcsize(VOCAB_ENTRY_PREFIX_FORMAT)


@dataclass(frozen=True)
class VocabularyEntry:
    token: bytes
    score: float


class Tokenizer:
    def __init__(
        self,
        vocab: list[bytes],
        scores: list[float],
        max_token_length: int,
        bos_id: int = 1,
        eos_id: int = 2,
        unk_id: int = 0,
    ) -> None:
        if len(vocab) != len(scores):
            raise ValueError(
                f"vocab and scores length mismatch: {len(vocab)} != {len(scores)}"
            )

        self.vocab = vocab
        self.scores = scores
        self.max_token_length = max_token_length
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.byte_to_id = {
            token[0]: idx for idx, token in enumerate(vocab) if len(token) == 1
        }
        self.sorted_vocabulary = sorted((token, idx) for idx, token in enumerate(vocab))

    @classmethod
    def from_file(
        cls,
        tokenizer_path: str | Path,
        vocab_size: int,
        bos_id: int = 1,
        eos_id: int = 2,
        unk_id: int = 0,
    ) -> "Tokenizer":
        path = Path(tokenizer_path)
        with path.open("rb") as fh:
            header = fh.read(TOKENIZER_HEADER_SIZE)
            if len(header) != TOKENIZER_HEADER_SIZE:
                raise ValueError("tokenizer header is truncated")

            (max_token_length,) = struct.unpack(TOKENIZER_HEADER_FORMAT, header)
            vocab: list[bytes] = []
            scores: list[float] = []

            for _ in range(vocab_size):
                entry_prefix = fh.read(VOCAB_ENTRY_PREFIX_SIZE)
                if len(entry_prefix) != VOCAB_ENTRY_PREFIX_SIZE:
                    raise ValueError("tokenizer entry prefix is truncated")

                score, length = struct.unpack(VOCAB_ENTRY_PREFIX_FORMAT, entry_prefix)
                if length < 0:
                    raise ValueError(f"token length must be non-negative, got {length}")

                token = fh.read(length)
                if len(token) != length:
                    raise ValueError("tokenizer entry token bytes are truncated")

                vocab.append(token)
                scores.append(score)

        return cls(
            vocab=vocab,
            scores=scores,
            max_token_length=max_token_length,
            bos_id=bos_id,
            eos_id=eos_id,
            unk_id=unk_id,
        )

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
        dummy_prefix: bool = False,
    ) -> list[int]:
        if dummy_prefix and text:
            text = " " + text

        pieces = self._initial_pieces(text)
        token_ids = [self._piece_to_token_id(piece) for piece in pieces]
        token_ids = self._merge_pairs(token_ids)

        if add_bos:
            token_ids.insert(0, self.bos_id)
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        buffer = bytearray()
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.vocab):
                raise ValueError(f"token id out of range: {token_id}")

            if skip_special_tokens and token_id in (self.bos_id, self.eos_id):
                continue

            buffer.extend(self.vocab[token_id])

        return bytes(buffer).decode("utf-8", errors="replace")

    def token_to_bytes(self, token_id: int) -> bytes:
        if token_id < 0 or token_id >= len(self.vocab):
            raise ValueError(f"token id out of range: {token_id}")
        return self.vocab[token_id]

    def _initial_pieces(self, text: str) -> list[bytes]:
        pieces: list[bytes] = []
        for char in text:
            encoded = char.encode("utf-8")
            if encoded in self.token_to_id:
                pieces.append(encoded)
                continue

            for byte_value in encoded:
                pieces.append(bytes([byte_value]))
        return pieces

    def _piece_to_token_id(self, piece: bytes) -> int:
        token_id = self.token_to_id.get(piece)
        if token_id is not None:
            return token_id

        if len(piece) == 1:
            byte_token_id = self.byte_to_id.get(piece[0])
            if byte_token_id is not None:
                return byte_token_id

        if self.unk_id < 0 or self.unk_id >= len(self.vocab):
            raise ValueError(f"unable to encode piece and unk_id is invalid: {piece!r}")
        return self.unk_id

    def _merge_pairs(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) < 2:
            return token_ids

        merged = token_ids[:]
        while True:
            best_score = None
            best_index = -1
            best_token_id = -1

            for index in range(len(merged) - 1):
                left = self.vocab[merged[index]]
                right = self.vocab[merged[index + 1]]
                candidate = left + right

                token_id = self.token_to_id.get(candidate)
                if token_id is None:
                    continue

                score = self.scores[token_id]
                if best_score is None or score > best_score:
                    best_score = score
                    best_index = index
                    best_token_id = token_id

            if best_index < 0:
                break

            merged[best_index : best_index + 2] = [best_token_id]

        return merged
