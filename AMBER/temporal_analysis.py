"""
Temporal analysis metrics for SOM classification results.

TemporalAnalysis takes a completed Classification object and computes
metrics that are only meaningful when the classified patterns form an
ordered time series:

  transition_matrix  — how often the SOM moves from neuron i to neuron j
  stability          — fraction of steps where the BMU does not change
  mean_path_length   — average Euclidean grid distance per step
  mean_chebyshev_jump — average Chebyshev grid distance per step
  temporal_coherence — fraction of steps with Chebyshev jump ≤ 1
  trajectory         — ordered sequence of (row, col) BMU positions
"""

import numpy as np


class TemporalAnalysis:
    """Temporal dynamics of a SOM classification result.

    Parameters
    ----------
    classification : Classification
        A completed Classification instance.  The patterns in
        classification.classification_data are assumed to be ordered
        in time (as they would be for windowed biosignals or audio).

    Attributes
    ----------
    trajectory : list of (int, int)
        Ordered BMU positions [(row_0, col_0), (row_1, col_1), ...].
    transition_matrix : ndarray, shape (n_neurons, n_neurons)
        Raw count of transitions; entry [i, j] = number of times the
        SOM moved from neuron i to neuron j between consecutive patterns.
        Neurons are linearised as  index = row * map_size + col.
    transition_matrix_norm : ndarray, shape (n_neurons, n_neurons)
        Row-normalised transition matrix (transition probabilities).
    stability : float
        Fraction of consecutive pattern pairs that share the same BMU.
    mean_path_length : float
        Mean Euclidean grid distance between consecutive BMU positions.
    mean_chebyshev_jump : float
        Mean Chebyshev (L∞) grid distance between consecutive BMU positions.
        Chebyshev distance counts diagonal steps as 1, matching the 8-neighbour
        topology of the SOM grid.
    temporal_coherence : float
        Fraction of consecutive pattern pairs whose BMUs are the same neuron or
        immediate neighbours (Chebyshev distance ≤ 1).  A value of 1.0 means
        every step stays within the local neighbourhood; a standard SOM on
        non-stationary data typically achieves 0.4–0.6, while a well-tuned
        RSOM approaches 1.0 on smooth temporal sequences.
    """

    def __init__(self, classification):
        cm = classification.classification_map
        self.map_size = classification.activations_map.shape[0]
        n_neurons = self.map_size ** 2

        # Build ordered trajectory
        self.trajectory = [
            (int(cm['x'].iloc[i]), int(cm['y'].iloc[i]))
            for i in range(len(cm))
        ]
        n = len(self.trajectory)

        # Transition matrix
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

        # Stability
        if n > 1:
            same = sum(
                1 for t in range(n - 1)
                if self.trajectory[t] == self.trajectory[t + 1]
            )
            self.stability = same / (n - 1)
        else:
            self.stability = 1.0

        # Mean path length (Euclidean grid distance)
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

        # Chebyshev (L∞) jump distances and Temporal Coherence
        # TC = fraction of consecutive steps with Chebyshev distance ≤ 1,
        # i.e. the BMU stays in the same neuron or moves to an immediate
        # neighbour (including diagonals) — matching the 8-neighbour SOM grid.
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
        """Return the top-k most frequent transitions as a list of dicts.

        Each dict has keys 'from' (row, col), 'to' (row, col), 'count'.
        """
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
        """Return a dict mapping each BMU (row, col) to its mean consecutive
        dwell time (number of steps the SOM stays on that neuron)."""
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

    def summary(self):
        """Print a short human-readable summary."""
        print(f"Sequence length     : {len(self.trajectory)}")
        print(f"Unique BMUs visited : {len(set(self.trajectory))}")
        print(f"Stability           : {self.stability:.3f}")
        print(f"Mean path length    : {self.mean_path_length:.3f} grid units (Euclidean)")
        print(f"Mean Chebyshev jump : {self.mean_chebyshev_jump:.3f} grid units")
        print(f"Temporal Coherence  : {self.temporal_coherence:.3f}  "
              f"(fraction of steps with Chebyshev jump ≤ 1)")
        top = self.most_frequent_transitions(3)
        print("Top-3 transitions   :")
        for tr in top:
            print(f"  {tr['from']} → {tr['to']}  ({tr['count']} times)")
