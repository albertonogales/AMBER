"""
Temporal metrics for SOM classification results assumed to be an ordered time series.

Computes transition_matrix, stability, mean_path_length, mean_chebyshev_jump,
temporal_coherence, and trajectory from a completed Classification object.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class TemporalAnalysis:
    """Temporal dynamics of a SOM classification result.

    Assumes classification patterns are time-ordered (e.g. windowed biosignals).
    Key attributes: trajectory, transition_matrix, stability, mean_path_length,
    mean_chebyshev_jump, temporal_coherence.
    """

    def __init__(self, classification):
        cm = classification.classification_map
        self.map_size = classification.activations_map.shape[0]
        n_neurons = self.map_size ** 2

        self.trajectory = [
            (int(cm['x'].iloc[i]), int(cm['y'].iloc[i]))
            for i in range(len(cm))
        ]
        n = len(self.trajectory)

        T = np.zeros((n_neurons, n_neurons), dtype=int)
        for t in range(n - 1):
            i = self.trajectory[t][0]     * self.map_size + self.trajectory[t][1]
            j = self.trajectory[t + 1][0] * self.map_size + self.trajectory[t + 1][1]
            T[i, j] += 1
        self.transition_matrix = T

        row_sums = T.sum(axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            self.transition_matrix_norm = np.where(
                row_sums > 0, T / row_sums, 0.0
            )

        if n > 1:
            same = sum(
                1 for t in range(n - 1)
                if self.trajectory[t] == self.trajectory[t + 1]
            )
            self.stability = same / (n - 1)
        else:
            self.stability = 1.0

        if n > 1:
            dists = [
                np.sqrt(
                    (self.trajectory[t][0] - self.trajectory[t + 1][0]) ** 2
                    + (self.trajectory[t][1] - self.trajectory[t + 1][1]) ** 2
                )
                for t in range(n - 1)
            ]
            self.mean_path_length = float(np.mean(dists))
        else:
            self.mean_path_length = 0.0

        # Temporal coherence = fraction of steps with Chebyshev jump ≤ 1 (same or immediate neighbour).
        if n > 1:
            chebyshev_jumps = [
                max(abs(self.trajectory[t][0] - self.trajectory[t + 1][0]),
                    abs(self.trajectory[t][1] - self.trajectory[t + 1][1]))
                for t in range(n - 1)
            ]
            self.mean_chebyshev_jump = float(np.mean(chebyshev_jumps))
            self.temporal_coherence  = float(
                sum(j <= 1 for j in chebyshev_jumps) / (n - 1)
            )
        else:
            self.mean_chebyshev_jump = 0.0
            self.temporal_coherence  = 1.0

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    def most_frequent_transitions(self, top_k=10):
        """Top-k transitions as list of dicts with keys 'from', 'to', 'count'."""
        flat = [
            (i // self.map_size, i % self.map_size,
             j // self.map_size, j % self.map_size,
             self.transition_matrix[i, j])
            for i in range(self.transition_matrix.shape[0])
            for j in range(self.transition_matrix.shape[1])
            if self.transition_matrix[i, j] > 0
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
