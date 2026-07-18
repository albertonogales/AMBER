"""
Signal-space and grid-space distance functions for AMBER SOMs.

Signal distances compare weight vectors to input patterns (BMU search);
grid distances compare 2-D neuron positions (neighbourhood update).
DTW uses L2 squared local cost + sqrt, equivalent to tslearn.metrics.dtw
(Berndt & Clifford, 1994).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict

import numpy as np

_DTW_WARN_THRESHOLD = 500  # samples; exposed so callers can suppress selectively

MatrixDistFn = Callable[..., Any]
ScalarDistFn = Callable[..., Any]

def euclidean_distance(a, b):
    """L2 distance between two 1-D arrays."""
    return np.sqrt(np.sum((a - b) ** 2))


def manhattan_distance(a, b):
    """L1 distance. More robust to spike artefacts than L2."""
    return np.sum(np.abs(a - b))


def chebyshev_distance(a, b):
    """L∞ distance (maximum absolute component difference)."""
    return np.max(np.abs(a - b))


def cosine_distance(a, b):
    """1 - cosine similarity. Amplitude-invariant; suited to spectral feature vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - np.dot(a, b) / (norm_a * norm_b)


def correlation_distance(a, b):
    """1 - |Pearson correlation|. Shape similarity only; ignores mean and amplitude."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    norm_a = np.linalg.norm(a_c)
    norm_b = np.linalg.norm(b_c)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - abs(np.dot(a_c, b_c) / (norm_a * norm_b))


def dtw_distance(a, b, band=None):
    """L2 DTW distance. Equals Euclidean when the warping path is the identity.

    :param band: Sakoe-Chiba half-width in samples (None = unconstrained).

    Pure-Python O(N·M) per pair — slow for windows >500 samples inside a SOM loop.
    Pass ``band`` (e.g. ``band=50``) to reduce cost to O(N·band).
    """
    n, m = len(a), len(b)
    if band is None and max(n, m) > _DTW_WARN_THRESHOLD:
        warnings.warn(
            f"dtw_distance: sequence length {max(n, m)} exceeds {_DTW_WARN_THRESHOLD} "
            f"samples. Pure-Python DTW is O(N²) per pair; inside a SOM training loop "
            f"this will be extremely slow. Pass band=<int> (e.g. band=50) to use a "
            f"Sakoe-Chiba corridor and reduce cost to O(N·band).",
            RuntimeWarning,
            stacklevel=2,
        )
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = 1 if band is None else max(1, i - band)
        j_end   = m if band is None else min(m, i + band)
        for j in range(j_start, j_end + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],
                dtw_matrix[i - 1, j - 1],
            )

    return np.sqrt(dtw_matrix[n, m])


def cross_correlation_distance(a, b):
    """1 - peak normalised cross-correlation. Shift-invariant; distance in [0, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 and norm_b == 0:
        return 0.0  # both zero → identical
    if norm_a == 0 or norm_b == 0:
        return 1.0  # one zero → maximally dissimilar
    a_n = a / norm_a
    b_n = b / norm_b
    xcorr = np.correlate(a_n, b_n, mode='full')
    return float(max(0.0, 1.0 - np.max(np.abs(xcorr))))


def euclidean_distance_matrix(weights, pattern):
    """(rows, cols) L2 distances from every neuron weight to pattern."""
    return np.sqrt(np.sum((weights - pattern) ** 2, axis=-1))


def manhattan_distance_matrix(weights, pattern):
    """(rows, cols) L1 distances."""
    return np.sum(np.abs(weights - pattern), axis=-1)


def chebyshev_distance_matrix(weights, pattern):
    """(rows, cols) L∞ distances."""
    return np.max(np.abs(weights - pattern), axis=-1)


def cosine_distance_matrix(weights, pattern):
    """(rows, cols) cosine distances. Vectorised over the full grid."""
    rows, cols, dim = weights.shape
    w_flat = weights.reshape(-1, dim)
    norms_w = np.linalg.norm(w_flat, axis=1)
    norm_p  = np.linalg.norm(pattern)
    denom = norms_w * norm_p
    with np.errstate(invalid='ignore', divide='ignore'):
        dots = w_flat @ pattern
        dist = np.where(denom == 0, 1.0, 1.0 - dots / denom)
    return dist.reshape(rows, cols)


def correlation_distance_matrix(weights, pattern):
    """(rows, cols) correlation distances. Vectorised."""
    rows, cols, dim = weights.shape
    w_flat = weights.reshape(-1, dim)
    w_centered = w_flat - w_flat.mean(axis=1, keepdims=True)
    p_centered = pattern - pattern.mean()
    norms_w = np.linalg.norm(w_centered, axis=1)
    norm_p  = np.linalg.norm(p_centered)
    with np.errstate(invalid='ignore', divide='ignore'):
        dots = w_centered @ p_centered
        denom = norms_w * norm_p
        dist = np.where(denom == 0, 1.0, 1.0 - np.abs(dots / denom))
    return dist.reshape(rows, cols)


def dtw_distance_matrix(weights, pattern, band=None):
    """(rows, cols) L2 DTW distances. Requires a per-neuron loop (sequential nature of DTW)."""
    rows, cols, _ = weights.shape
    dist = np.empty((rows, cols))
    for i in range(rows):
        for j in range(cols):
            dist[i, j] = dtw_distance(weights[i, j], pattern, band=band)
    return dist


def cross_correlation_distance_matrix(weights, pattern):
    """(rows, cols) cross-correlation distances. Requires a per-neuron loop."""
    rows, cols, _ = weights.shape
    dist = np.empty((rows, cols))
    for i in range(rows):
        for j in range(cols):
            dist[i, j] = cross_correlation_distance(weights[i, j], pattern)
    return dist


def grid_euclidean(ids_matrix, bmu):
    """Euclidean distance from every grid position to bmu. Returns (rows, cols) array."""
    return np.sqrt(np.sum(np.square(ids_matrix - np.array(bmu)), axis=-1))


def grid_chebyshev(ids_matrix, bmu):
    """Chebyshev distance from every grid position to bmu. Returns (rows, cols) array."""
    return np.max(np.abs(ids_matrix - np.array(bmu)), axis=-1)


SIGNAL_DISTANCE_MATRIX: Dict[str, MatrixDistFn] = {
    'euclidean':         euclidean_distance_matrix,
    'manhattan':         manhattan_distance_matrix,
    'chebyshev':         chebyshev_distance_matrix,
    'cosine':            cosine_distance_matrix,
    'correlation':       correlation_distance_matrix,
    'dtw':               dtw_distance_matrix,
    'cross_correlation': cross_correlation_distance_matrix,
}

SIGNAL_DISTANCE_SCALAR: Dict[str, ScalarDistFn] = {
    'euclidean':         euclidean_distance,
    'manhattan':         manhattan_distance,
    'chebyshev':         chebyshev_distance,
    'cosine':            cosine_distance,
    'correlation':       correlation_distance,
    'dtw':               dtw_distance,
    'cross_correlation': cross_correlation_distance,
}

GRID_DISTANCE = {
    'euclidean': grid_euclidean,
    'chebyshev': grid_chebyshev,
}

AVAILABLE_DISTANCES = list(SIGNAL_DISTANCE_MATRIX.keys())
