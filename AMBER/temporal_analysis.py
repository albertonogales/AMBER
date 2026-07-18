"""Temporal metrics for a time-ordered SOM classification result."""

import logging
from functools import cached_property

import numpy as np

logger = logging.getLogger(__name__)


class TemporalAnalysis:
    """Temporal dynamics of a time-ordered SOM classification result.

    Assumes classification patterns are ordered (e.g. windowed biosignals).
    All metrics are computed lazily on first access via @cached_property.
    """

    def __init__(self, classification) -> None:
        cm = classification.classification_map
        self.map_size = classification.activations_map.shape[0]

        xs = cm['x'].to_numpy(dtype=int)
        ys = cm['y'].to_numpy(dtype=int)
        self.trajectory = list(zip(xs.tolist(), ys.tolist()))

        self._xs = xs
        self._ys = ys

    @cached_property
    def transition_matrix(self) -> np.ndarray:
        """k²×k² count matrix; T[i,j] = number of steps from neuron i to neuron j."""
        n_neurons = self.map_size ** 2
        T = np.zeros((n_neurons, n_neurons), dtype=int)
        indices = self._xs * self.map_size + self._ys
        if len(indices) > 1:
            np.add.at(T, (indices[:-1], indices[1:]), 1)
        return T

    @cached_property
    def transition_matrix_norm(self) -> np.ndarray:
        """Row-normalised transition matrix (transition probabilities)."""
        T = self.transition_matrix
        row_sums = T.sum(axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(row_sums > 0, T / row_sums, 0.0)

    @cached_property
    def stability(self) -> float:
        """Fraction of consecutive steps where the BMU did not change."""
        n = len(self.trajectory)
        if n <= 1:
            return 1.0
        same = np.sum((self._xs[:-1] == self._xs[1:]) & (self._ys[:-1] == self._ys[1:]))
        return float(same / (n - 1))

    @cached_property
    def mean_path_length(self) -> float:
        """Mean Euclidean distance between consecutive BMU positions (grid units)."""
        n = len(self.trajectory)
        if n <= 1:
            return 0.0
        dx = np.diff(self._xs).astype(float)
        dy = np.diff(self._ys).astype(float)
        return float(np.mean(np.sqrt(dx ** 2 + dy ** 2)))

    @cached_property
    def mean_chebyshev_jump(self) -> float:
        """Mean Chebyshev distance between consecutive BMU positions (grid units)."""
        n = len(self.trajectory)
        if n <= 1:
            return 0.0
        return float(np.mean(np.maximum(np.abs(np.diff(self._xs)),
                                        np.abs(np.diff(self._ys)))))

    @cached_property
    def temporal_coherence(self) -> float:
        """Fraction of steps where Chebyshev jump ≤ 1 (same neuron or immediate neighbour)."""
        n = len(self.trajectory)
        if n <= 1:
            return 1.0
        jumps = np.maximum(np.abs(np.diff(self._xs)), np.abs(np.diff(self._ys)))
        return float(np.sum(jumps <= 1) / (n - 1))

    def most_frequent_transitions(self, top_k: int = 10):
        """Top-k transitions as list of dicts with keys 'from', 'to', 'count'."""
        T = self.transition_matrix
        flat = [
            (i // self.map_size, i % self.map_size,
             j // self.map_size, j % self.map_size,
             T[i, j])
            for i in range(T.shape[0])
            for j in range(T.shape[1])
            if T[i, j] > 0
        ]
        flat.sort(key=lambda x: -x[4])
        return [
            {'from': (r1, c1), 'to': (r2, c2), 'count': cnt}
            for r1, c1, r2, c2, cnt in flat[:top_k]
        ]

    def dwell_times(self):
        """Mean consecutive dwell time (steps) per BMU (row, col)."""
        dwell = {}
        t = 0
        n = len(self.trajectory)
        while t < n:
            pos = self.trajectory[t]
            run = 1
            while t + run < n and self.trajectory[t + run] == pos:
                run += 1
            if pos not in dwell:
                dwell[pos] = []
            dwell[pos].append(run)
            t += run
        return {pos: float(np.mean(runs)) for pos, runs in dwell.items()}

    def summary(self) -> str:
        """Human-readable summary of temporal metrics; also logged at INFO."""
        top = self.most_frequent_transitions(3)
        transitions = "\n".join(
            f"  {tr['from']} → {tr['to']}  ({tr['count']} times)" for tr in top
        )
        text = (
            f"Sequence length     : {len(self.trajectory)}\n"
            f"Unique BMUs visited : {len(set(self.trajectory))}\n"
            f"Stability           : {self.stability:.3f}\n"
            f"Mean path length    : {self.mean_path_length:.3f} grid units (Euclidean)\n"
            f"Mean Chebyshev jump : {self.mean_chebyshev_jump:.3f} grid units\n"
            f"Temporal Coherence  : {self.temporal_coherence:.3f}  "
            f"(fraction of steps with Chebyshev jump ≤ 1)\n"
            f"Top-3 transitions   :\n{transitions}"
        )
        logger.info(text)
        return text
