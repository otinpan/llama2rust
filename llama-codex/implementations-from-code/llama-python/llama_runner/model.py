from __future__ import annotations

from pathlib import Path
import numpy as np

from .config import Config
from .kernels import matmul, rmsnorm, silu, softmax
from .state import RunState
from .weights import Checkpoint, TransformerWeights, load_checkpoint


class Transformer:
    def __init__(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint
        self.config: Config = checkpoint.config
        self.weights: TransformerWeights = checkpoint.weights
        self.state = RunState.create(self.config)

    @classmethod
    def from_file(cls, path: str | Path) -> "Transformer":
        return cls(load_checkpoint(path))

    def close(self) -> None:
        self.checkpoint.close()

    def reset_state(self) -> None:
        self.state = RunState.create(self.config)

    def forward(self, token: int, pos: int) -> np.ndarray:
        p = self.config
        w = self.weights
        s = self.state

        dim = p.dim
        hidden_dim = p.hidden_dim
        head_size = dim // p.n_heads
        kv_dim = (dim * p.n_kv_heads) // p.n_heads
        kv_mul = p.n_heads // p.n_kv_heads

        s.x[:] = w.token_embedding_table[token]

        for layer in range(p.n_layers):
            s.xb[:] = rmsnorm(s.x, w.rms_att_weight[layer])

            s.q[:] = matmul(w.wq[layer], s.xb)
            s.k[:] = matmul(w.wk[layer], s.xb)
            s.v[:] = matmul(w.wv[layer], s.xb)

            s.key_cache[layer, pos, :] = s.k
            s.value_cache[layer, pos, :] = s.v

            for i in range(0, dim, 2):
                head_dim = i % head_size
                freq = np.float32(1.0 / (10000.0 ** (head_dim / head_size)))
                angle = np.float32(pos) * freq
                fcr = np.float32(np.cos(angle))
                fci = np.float32(np.sin(angle))
                rotate_count = 2 if i < kv_dim else 1
                for vec in range(rotate_count):
                    target = s.q if vec == 0 else s.k
                    v0 = target[i]
                    v1 = target[i + 1]
                    target[i] = v0 * fcr - v1 * fci
                    target[i + 1] = v0 * fci + v1 * fcr

            for head in range(p.n_heads):
                q = s.q[head * head_size : (head + 1) * head_size]
                att = s.att[head, : pos + 1]
                for t in range(pos + 1):
                    k = s.key_cache[layer, t, (head // kv_mul) * head_size : (head // kv_mul + 1) * head_size]
                    att[t] = np.dot(q, k) / np.float32(np.sqrt(head_size))
                att[:] = softmax(att)

                xb_head = s.xb[head * head_size : (head + 1) * head_size]
                xb_head.fill(0.0)
                for t in range(pos + 1):
                    v = s.value_cache[layer, t, (head // kv_mul) * head_size : (head // kv_mul + 1) * head_size]
                    xb_head += att[t] * v

            s.xb2[:] = matmul(w.wo[layer], s.xb)
            s.x += s.xb2

            s.xb[:] = rmsnorm(s.x, w.rms_ffn_weight[layer])
            s.hb[:] = matmul(w.w1[layer], s.xb)
            s.hb2[:] = matmul(w.w3[layer], s.xb)
            s.hb[:] = silu(s.hb) * s.hb2
            s.xb[:] = matmul(w.w2[layer], s.hb)
            s.x += s.xb

        s.x[:] = rmsnorm(s.x, w.rms_final_weight)
        s.logits[:] = matmul(w.wcls, s.x)
        return s.logits.copy()
