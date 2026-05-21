from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .config import Config


@dataclass(slots=True)
class TransformerWeights:
    token_embedding_table: np.ndarray
    rms_att_weight: np.ndarray
    rms_ffn_weight: np.ndarray
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray
    rms_final_weight: np.ndarray
    wcls: np.ndarray


@dataclass(slots=True)
class Checkpoint:
    config: Config
    weights: TransformerWeights
    shared_weights: bool
    raw_floats: np.ndarray

    def close(self) -> None:
        # We keep weights as copied NumPy arrays, so there is nothing to release here.
        return


def _take(raw: np.ndarray, offset: int, size: int, shape: tuple[int, ...]) -> tuple[np.ndarray, int]:
    next_offset = offset + size
    view = raw[offset:next_offset].reshape(shape)
    return view, next_offset


def load_checkpoint(path: str | Path) -> Checkpoint:
    checkpoint_path = Path(path)
    with checkpoint_path.open("rb") as handle:
        header = handle.read(Config.byte_size())
        config = Config.from_bytes(header)
        shared_weights = config.vocab_size > 0
        vocab_size = abs(config.vocab_size)
        config = Config(
            dim=config.dim,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            vocab_size=vocab_size,
            seq_len=config.seq_len,
        )
        handle.seek(Config.byte_size())
        raw_floats = np.frombuffer(handle.read(), dtype=np.float32).copy()

    head_size = config.dim // config.n_heads
    kv_dim = (config.dim * config.n_kv_heads) // config.n_heads
    offset = 0

    token_embedding_table, offset = _take(
        raw_floats, offset, config.vocab_size * config.dim, (config.vocab_size, config.dim)
    )
    rms_att_weight, offset = _take(
        raw_floats, offset, config.n_layers * config.dim, (config.n_layers, config.dim)
    )
    wq, offset = _take(
        raw_floats, offset, config.n_layers * config.dim * config.dim, (config.n_layers, config.dim, config.dim)
    )
    wk, offset = _take(
        raw_floats, offset, config.n_layers * config.dim * kv_dim, (config.n_layers, kv_dim, config.dim)
    )
    wv, offset = _take(
        raw_floats, offset, config.n_layers * config.dim * kv_dim, (config.n_layers, kv_dim, config.dim)
    )
    wo, offset = _take(
        raw_floats, offset, config.n_layers * config.dim * config.dim, (config.n_layers, config.dim, config.dim)
    )
    rms_ffn_weight, offset = _take(
        raw_floats, offset, config.n_layers * config.dim, (config.n_layers, config.dim)
    )
    w1, offset = _take(
        raw_floats,
        offset,
        config.n_layers * config.dim * config.hidden_dim,
        (config.n_layers, config.hidden_dim, config.dim),
    )
    w2, offset = _take(
        raw_floats,
        offset,
        config.n_layers * config.hidden_dim * config.dim,
        (config.n_layers, config.dim, config.hidden_dim),
    )
    w3, offset = _take(
        raw_floats,
        offset,
        config.n_layers * config.dim * config.hidden_dim,
        (config.n_layers, config.hidden_dim, config.dim),
    )
    rms_final_weight, offset = _take(raw_floats, offset, config.dim, (config.dim,))

    rope_skip = config.seq_len * head_size // 2
    offset += rope_skip
    offset += rope_skip

    if shared_weights:
        wcls = token_embedding_table
    else:
        wcls, offset = _take(raw_floats, offset, config.vocab_size * config.dim, (config.vocab_size, config.dim))

    weights = TransformerWeights(
        token_embedding_table=token_embedding_table,
        rms_att_weight=rms_att_weight,
        rms_ffn_weight=rms_ffn_weight,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
        w1=w1,
        w2=w2,
        w3=w3,
        rms_final_weight=rms_final_weight,
        wcls=wcls,
    )
    return Checkpoint(
        config=config,
        weights=weights,
        shared_weights=shared_weights,
        raw_floats=raw_floats,
    )
