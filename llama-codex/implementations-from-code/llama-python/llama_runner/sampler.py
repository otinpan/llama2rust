from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from .kernels import softmax


def random_u32(state: int) -> tuple[int, int]:
    state ^= state >> 12
    state ^= (state << 25) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 27
    value = ((state * 0x2545F4914F6CDD1D) & 0xFFFFFFFFFFFFFFFF) >> 32
    return value, state


def random_f32(state: int) -> tuple[float, int]:
    value, next_state = random_u32(state)
    return ((value >> 8) / 16777216.0), next_state


@dataclass(slots=True)
class Sampler:
    vocab_size: int
    temperature: float = 1.0
    topp: float = 0.9
    rng_state: int = 0

    def __post_init__(self) -> None:
        if self.rng_state <= 0:
            self.rng_state = int(time.time()) & 0xFFFFFFFFFFFFFFFF

    def sample_argmax(self, probabilities: np.ndarray) -> int:
        return int(np.argmax(probabilities))

    def sample_mult(self, probabilities: np.ndarray, coin: float) -> int:
        cdf = np.cumsum(probabilities, dtype=np.float32)
        idx = int(np.searchsorted(cdf, coin, side="right"))
        return min(idx, self.vocab_size - 1)

    def sample_topp(self, probabilities: np.ndarray, coin: float) -> int:
        cutoff = (1.0 - self.topp) / (self.vocab_size - 1)
        candidates = np.flatnonzero(probabilities >= cutoff)
        if candidates.size == 0:
            return int(np.argmax(probabilities))

        candidate_probs = probabilities[candidates]
        order = np.argsort(-candidate_probs)
        sorted_indices = candidates[order]
        sorted_probs = candidate_probs[order]

        cumulative = np.cumsum(sorted_probs, dtype=np.float32)
        last_idx = int(np.searchsorted(cumulative, self.topp, side="right"))
        last_idx = min(last_idx, len(sorted_indices) - 1)

        truncated_probs = sorted_probs[: last_idx + 1]
        truncated_sum = float(np.sum(truncated_probs, dtype=np.float32))
        r = coin * truncated_sum
        cdf = 0.0
        for idx, prob in zip(sorted_indices[: last_idx + 1], truncated_probs):
            cdf += float(prob)
            if r < cdf:
                return int(idx)
        return int(sorted_indices[last_idx])

    def sample(self, logits: np.ndarray) -> int:
        if self.temperature == 0.0:
            return self.sample_argmax(logits)

        probs = softmax(logits / self.temperature)
        coin, self.rng_state = random_f32(self.rng_state)
        if self.topp <= 0.0 or self.topp >= 1.0:
            return self.sample_mult(probs, coin)
        return self.sample_topp(probs, coin)
