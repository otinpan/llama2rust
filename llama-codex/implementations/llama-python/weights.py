# @trace-pilot a99b2ede2dbf1d4b2169ffb04b009d60ce04cfb2
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from config import Config, HEADER_SIZE


FLOAT32_SIZE = np.dtype(np.float32).itemsize


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]

    @property
    def size(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


@dataclass
class TransformerWeights:
    token_embedding_table: np.ndarray
    rms_att_weight: np.ndarray
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wo: np.ndarray
    rms_ffn_weight: np.ndarray
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray
    rms_final_weight: np.ndarray
    freq_cis_real: np.ndarray
    freq_cis_imag: np.ndarray
    wcls: np.ndarray
    shared_classifier: bool
    _mapped: np.memmap = field(repr=False)


def build_weight_specs(config: Config) -> tuple[list[TensorSpec], TensorSpec]:
    head_size = config.head_size
    if head_size % 2 != 0:
        raise ValueError(f"head_size must be even for RoPE, got {head_size}")

    kv_dim = config.kv_dim
    rope_shape = (config.seq_len, head_size // 2)

    specs = [
        TensorSpec("token_embedding_table", (config.vocab_size, config.dim)),
        TensorSpec("rms_att_weight", (config.n_layers, config.dim)),
        TensorSpec("wq", (config.n_layers, config.dim, config.dim)),
        TensorSpec("wk", (config.n_layers, config.dim, kv_dim)),
        TensorSpec("wv", (config.n_layers, config.dim, kv_dim)),
        TensorSpec("wo", (config.n_layers, config.dim, config.dim)),
        TensorSpec("rms_ffn_weight", (config.n_layers, config.dim)),
        TensorSpec("w1", (config.n_layers, config.hidden_dim, config.dim)),
        TensorSpec("w2", (config.n_layers, config.dim, config.hidden_dim)),
        TensorSpec("w3", (config.n_layers, config.hidden_dim, config.dim)),
        TensorSpec("rms_final_weight", (config.dim,)),
        TensorSpec("freq_cis_real", rope_shape),
        TensorSpec("freq_cis_imag", rope_shape),
    ]
    classifier_spec = TensorSpec("wcls", (config.vocab_size, config.dim))
    return specs, classifier_spec


def expected_weight_counts(config: Config) -> tuple[int, int]:
    specs, classifier_spec = build_weight_specs(config)
    shared_count = sum(spec.size for spec in specs)
    unshared_count = shared_count + classifier_spec.size
    return shared_count, unshared_count


def load_weights(model_path: str | Path, config: Config | None = None) -> TransformerWeights:
    path = Path(model_path)
    if config is None:
        config = Config.from_file(path)

    specs, classifier_spec = build_weight_specs(config)
    weights_count = _validate_file_size(path, config)
    mapped = np.memmap(
        path,
        dtype=np.float32,
        mode="r",
        offset=HEADER_SIZE,
        shape=(weights_count,),
    )

    offset = 0
    tensors: dict[str, np.ndarray] = {}
    for spec in specs:
        next_offset = offset + spec.size
        tensors[spec.name] = mapped[offset:next_offset].reshape(spec.shape)
        offset = next_offset

    shared_classifier = offset == weights_count
    if shared_classifier:
        tensors["wcls"] = tensors["token_embedding_table"]
    else:
        next_offset = offset + classifier_spec.size
        tensors["wcls"] = mapped[offset:next_offset].reshape(classifier_spec.shape)
        offset = next_offset

    if offset != weights_count:
        raise ValueError(
            f"invalid final weight offset: consumed {offset} floats, file has {weights_count}"
        )

    return TransformerWeights(
        token_embedding_table=tensors["token_embedding_table"],
        rms_att_weight=tensors["rms_att_weight"],
        wq=tensors["wq"],
        wk=tensors["wk"],
        wv=tensors["wv"],
        wo=tensors["wo"],
        rms_ffn_weight=tensors["rms_ffn_weight"],
        w1=tensors["w1"],
        w2=tensors["w2"],
        w3=tensors["w3"],
        rms_final_weight=tensors["rms_final_weight"],
        freq_cis_real=tensors["freq_cis_real"],
        freq_cis_imag=tensors["freq_cis_imag"],
        wcls=tensors["wcls"],
        shared_classifier=shared_classifier,
        _mapped=mapped,
    )


def _validate_file_size(path: Path, config: Config) -> int:
    file_size = path.stat().st_size
    if file_size < HEADER_SIZE:
        raise ValueError(
            f"model file is smaller than header: size={file_size}, header={HEADER_SIZE}"
        )

    payload_size = file_size - HEADER_SIZE
    if payload_size % FLOAT32_SIZE != 0:
        raise ValueError(
            f"weight payload size must be a multiple of {FLOAT32_SIZE}, got {payload_size}"
        )

    weights_count = payload_size // FLOAT32_SIZE
    shared_count, unshared_count = expected_weight_counts(config)
    if weights_count not in (shared_count, unshared_count):
        raise ValueError(
            "unexpected weight count: "
            f"got {weights_count}, expected {shared_count} (shared wcls) "
            f"or {unshared_count} (separate wcls)"
        )

    return weights_count
