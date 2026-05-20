# @trace-pilot c5740e0fa9f704c4479b5ee4e8515a927cfb0938
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import Config
from weights import TransformerWeights


RMS_NORM_EPS = 1e-5


@dataclass
class RunState:
    x: np.ndarray
    xb: np.ndarray
    xb2: np.ndarray
    hb: np.ndarray
    hb2: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    att: np.ndarray
    logits: np.ndarray
    key_cache: np.ndarray
    value_cache: np.ndarray


def create_run_state(config: Config) -> RunState:
    dim = config.dim
    hidden_dim = config.hidden_dim
    kv_dim = config.kv_dim

    return RunState(
        x=np.zeros(dim, dtype=np.float32),
        xb=np.zeros(dim, dtype=np.float32),
        xb2=np.zeros(dim, dtype=np.float32),
        hb=np.zeros(hidden_dim, dtype=np.float32),
        hb2=np.zeros(hidden_dim, dtype=np.float32),
        q=np.zeros(dim, dtype=np.float32),
        k=np.zeros(kv_dim, dtype=np.float32),
        v=np.zeros(kv_dim, dtype=np.float32),
        att=np.zeros((config.n_heads, config.seq_len), dtype=np.float32),
        logits=np.zeros(config.vocab_size, dtype=np.float32),
        key_cache=np.zeros((config.n_layers, config.seq_len, kv_dim), dtype=np.float32),
        value_cache=np.zeros((config.n_layers, config.seq_len, kv_dim), dtype=np.float32),
    )


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float = RMS_NORM_EPS) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(x), dtype=np.float32) + eps)
    return (x / rms) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    freq_real: np.ndarray,
    freq_imag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    q_out = q.copy()
    k_out = k.copy()
    _apply_rope_inplace(q_out, freq_real, freq_imag)
    _apply_rope_inplace(k_out, freq_real, freq_imag)
    return q_out, k_out


def _apply_rope_inplace(x: np.ndarray, freq_real: np.ndarray, freq_imag: np.ndarray) -> None:
    for head in range(x.shape[0]):
        for pair_idx in range(x.shape[1] // 2):
            i0 = 2 * pair_idx
            i1 = i0 + 1
            real = x[head, i0]
            imag = x[head, i1]
            cos = freq_real[pair_idx]
            sin = freq_imag[pair_idx]
            x[head, i0] = real * cos - imag * sin
            x[head, i1] = real * sin + imag * cos


class Transformer:
    def __init__(
        self,
        config: Config,
        weights: TransformerWeights,
        state: RunState | None = None,
    ) -> None:
        self.config = config
        self.weights = weights
        self.state = state if state is not None else create_run_state(config)
        self.kv_mul = config.n_heads // config.n_kv_heads
        self.att_scale = 1.0 / np.sqrt(config.head_size)

    def forward(self, token: int, pos: int) -> np.ndarray:
        if token < 0 or token >= self.config.vocab_size:
            raise ValueError(f"token out of range: {token}")
        if pos < 0 or pos >= self.config.seq_len:
            raise ValueError(f"position out of range: {pos}")

        state = self.state
        weights = self.weights
        config = self.config

        state.x[...] = weights.token_embedding_table[token]

        for layer in range(config.n_layers):
            state.xb[...] = rmsnorm(state.x, weights.rms_att_weight[layer])

            state.q[...] = state.xb @ weights.wq[layer]
            state.k[...] = state.xb @ weights.wk[layer]
            state.v[...] = state.xb @ weights.wv[layer]

            q_heads = state.q.reshape(config.n_heads, config.head_size)
            k_heads = state.k.reshape(config.n_kv_heads, config.head_size)
            freq_real = weights.freq_cis_real[pos]
            freq_imag = weights.freq_cis_imag[pos]
            q_heads, k_heads = apply_rope(q_heads, k_heads, freq_real, freq_imag)

            state.q[...] = q_heads.reshape(config.dim)
            state.k[...] = k_heads.reshape(config.kv_dim)
            state.key_cache[layer, pos, :] = state.k
            state.value_cache[layer, pos, :] = state.v

            state.xb2.fill(0.0)
            keys = state.key_cache[layer, : pos + 1].reshape(
                pos + 1, config.n_kv_heads, config.head_size
            )
            values = state.value_cache[layer, : pos + 1].reshape(
                pos + 1, config.n_kv_heads, config.head_size
            )

            for head in range(config.n_heads):
                kv_head = head // self.kv_mul
                q_head = q_heads[head]
                scores = keys[:, kv_head, :] @ q_head
                scores *= self.att_scale
                probs = softmax(scores.astype(np.float32))
                state.att[head, : pos + 1] = probs
                state.att[head, pos + 1 :] = 0.0

                context = probs @ values[:, kv_head, :]
                start = head * config.head_size
                end = start + config.head_size
                state.xb2[start:end] = context

            state.xb[...] = state.xb2 @ weights.wo[layer]
            state.x[...] = state.x + state.xb

            state.xb[...] = rmsnorm(state.x, weights.rms_ffn_weight[layer])
            state.hb[...] = weights.w1[layer] @ state.xb
            state.hb2[...] = weights.w3[layer] @ state.xb
            state.hb[...] = silu(state.hb) * state.hb2
            state.xb[...] = weights.w2[layer] @ state.hb
            state.x[...] = state.x + state.xb

        state.x[...] = rmsnorm(state.x, weights.rms_final_weight)
        state.logits[...] = weights.wcls @ state.x
        return state.logits
