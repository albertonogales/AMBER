"""
Signal-space distance functions for AMBER.

Two families are provided:

  Signal distances  — compare weight vectors to an input pattern; used for BMU search.
  Grid distances    — compare 2-D neuron positions on the map grid; used for neighbourhood update.

Each signal distance exposes:
  - a scalar function  foo_distance(a, b)           for single pairs
  - a matrix function  foo_distance_matrix(W, p)    that returns a (rows, cols) array over the
                                                     whole weight grid W shaped (rows, cols, dim)

Vectorised matrix functions are provided for all distances except DTW and cross-correlation,
which require a per-neuron loop due to their sequential nature.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Signal distances — scalar
# ---------------------------------------------------------------------------

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
    """``1 - abs(Pearson correlation)``. Pure shape similarity; ignores mean and amplitude.
    Ideal for comparing waveform morphology across subjects or sessions."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    norm_a = np.linalg.norm(a_c)
    norm_b = np.linalg.norm(b_c)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - abs(np.dot(a_c, b_c) / (norm_a * norm_b))


def dtw_distance(a, b, band=None):
    """Dynamic Time Warping distance with optional Sakoe-Chiba band constraint.

    Handles temporal misalignment between signals — the standard choice for
    biosignals (ECG, EEG) and audio where patterns may be stretched or shifted
    in time.

    :param a: 1-D array, first signal
    :param b: 1-D array, second signal
    :param band: Sakoe-Chiba half-width in samples (None = unconstrained).
                 Constraining the band greatly reduces O(n²) cost; a value of
                 10–20 % of signal length is a good default for biosignals.
    :return: DTW distance (scalar)
    """
    n, m = len(a), len(b)
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
    """1 - peak of normalised cross-correlation. Shift-invariant similarity.

    Suitable for periodic biosignals (ECG beats, EEG oscillations) where the
    pattern of interest may appear at different phases across windows.

    Both inputs are L2-normalised before correlation.  By the Cauchy-Schwarz
    inequality every lag of ``np.correlate(a_n, b_n, 'full')`` is bounded by
    the product of the partial L2-norms of the two sub-vectors, which are each
    ≤ 1, so ``max|xcorr| ∈ [0, 1]`` and the returned distance is in ``[0, 1]``.

    :param a: 1-D array
    :param b: 1-D array
    :return: distance in [0, 1]; 0 means perfect match at some lag
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    a_n = a / norm_a
    b_n = b / norm_b
    xcorr = np.correlate(a_n, b_n, mode='full')
    return 1.0 - float(np.max(np.abs(xcorr)))


# ---------------------------------------------------------------------------
# Signal distances — matrix (whole weight grid vs. one pattern)
# ---------------------------------------------------------------------------

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
    w_flat = weights.reshape(-1, dim)          # (rows*cols, dim)
    norms_w = np.linalg.norm(w_flat, axis=1)  # (rows*cols,)
    norm_p  = np.linalg.norm(pattern)
    denom = norms_w * norm_p
    # where denominator is zero, distance is 1
    with np.errstate(invalid='ignore', divide='ignore'):
        dots = w_flat @ pattern                # (rows*cols,)
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
    """(rows, cols) DTW distances. Requires a per-neuron loop."""
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


# ---------------------------------------------------------------------------
# Grid distances — neuron position space (used for neighbourhood update)
# ---------------------------------------------------------------------------

def grid_euclidean(ids_matrix, bmu):
    """Euclidean distance from every grid position to bmu. Returns (rows, cols) array."""
    return np.sqrt(np.sum(np.square(ids_matrix - np.array(bmu)), axis=-1))


def grid_chebyshev(ids_matrix, bmu):
    """Chebyshev distance from every grid position to bmu. Returns (rows, cols) array."""
    return np.max(np.abs(ids_matrix - np.array(bmu)), axis=-1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SIGNAL_DISTANCE_MATRIX = {
    'euclidean':         euclidean_distance_matrix,
    'manhattan':         manhattan_distance_matrix,
    'chebyshev':         chebyshev_distance_matrix,
    'cosine':            cosine_distance_matrix,
    'correlation':       correlation_distance_matrix,
    'dtw':               dtw_distance_matrix,
    'cross_correlation': cross_correlation_distance_matrix,
}

SIGNAL_DISTANCE_SCALAR = {
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
