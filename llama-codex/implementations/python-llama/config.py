from __future__ import annotations

from dataclasses import dataclass
import struct
from pathlib import Path


HEADER_FORMAT = "<7i"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# test
@dataclass(frozen=True)
class Config:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.dim * self.n_kv_heads // self.n_heads

    @classmethod
    def from_header_bytes(cls, data: bytes) -> "Config":
        if len(data) != HEADER_SIZE:
            raise ValueError(
                f"invalid header size: expected {HEADER_SIZE} bytes, got {len(data)} bytes"
            )

        config = cls(*struct.unpack(HEADER_FORMAT, data))
        config.validate()
        return config

    @classmethod
    def from_file(cls, model_path: str | Path) -> "Config":
        path = Path(model_path)
        with path.open("rb") as fh:
            header = fh.read(HEADER_SIZE)
        return cls.from_header_bytes(header)

    def validate(self) -> None:
        fields = {
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
        }
        for name, value in fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        if self.dim % self.n_heads != 0:
            raise ValueError(
                f"dim must be divisible by n_heads: dim={self.dim}, n_heads={self.n_heads}"
            )

        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                "n_heads must be divisible by n_kv_heads: "
                f"n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}"
            )