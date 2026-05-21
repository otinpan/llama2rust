from __future__ import annotations

from dataclasses import dataclass
import struct


@dataclass(slots=True)
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int

    STRUCT = struct.Struct("7i")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Config":
        values = cls.STRUCT.unpack(data)
        return cls(*values)

    @classmethod
    def byte_size(cls) -> int:
        return cls.STRUCT.size
