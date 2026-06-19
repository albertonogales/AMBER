"""Tests for AMBER/distances.py — all signal and grid distance functions."""

import numpy as np
import pytest

from AMBER.distances import (
    euclidean_distance, euclidean_distance_matrix,
    manhattan_distance, manhattan_distance_matrix,
    chebyshev_distance, chebyshev_distance_matrix,
    cosine_distance, cosine_distance_matrix,
    correlation_distance, correlation_distance_matrix,
    dtw_distance, dtw_distance_matrix,
    cross_correlation_distance, cross_correlation_distance_matrix,
    grid_euclidean, grid_chebyshev,
    SIGNAL_DISTANCE_MATRIX, AVAILABLE_DISTANCES,
)

RNG = np.random.default_rng(99)
A = RNG.standard_normal(20)
B = RNG.standard_normal(20)
ZERO = np.zeros(20)
WEIGHTS = RNG.standard_normal((5, 5, 20))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_available_distances_list():
    expected = {'euclidean', 'manhattan', 'chebyshev', 'cosine',
                'correlation', 'dtw', 'cross_correlation'}
    assert set(AVAILABLE_DISTANCES) == expected


def test_registry_keys_match_available():
    assert set(SIGNAL_DISTANCE_MATRIX.keys()) == set(AVAILABLE_DISTANCES)


# ---------------------------------------------------------------------------
# Euclidean
# ---------------------------------------------------------------------------

def test_euclidean_identity():
    assert euclidean_distance(A, A) == pytest.approx(0.0)


def test_euclidean_symmetry():
    assert euclidean_distance(A, B) == pytest.approx(euclidean_distance(B, A))


def test_euclidean_non_negative():
    assert euclidean_distance(A, B) >= 0.0


def test_euclidean_known_value():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert euclidean_distance(a, b) == pytest.approx(5.0)


def test_euclidean_matrix_shape():
    d = euclidean_distance_matrix(WEIGHTS, A)
    assert d.shape == (5, 5)


def test_euclidean_matrix_matches_scalar():
    d = euclidean_distance_matrix(WEIGHTS, A)
    assert d[2, 3] == pytest.approx(euclidean_distance(WEIGHTS[2, 3], A))


def test_euclidean_matrix_identity():
    # If pattern equals one weight, that cell should be 0
    W = WEIGHTS.copy()
    W[1, 1] = A
    d = euclidean_distance_matrix(W, A)
    assert d[1, 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Manhattan
# ---------------------------------------------------------------------------

def test_manhattan_identity():
    assert manhattan_distance(A, A) == pytest.approx(0.0)


def test_manhattan_non_negative():
    assert manhattan_distance(A, B) >= 0.0


def test_manhattan_known_value():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 6.0, 8.0])
    assert manhattan_distance(a, b) == pytest.approx(12.0)


def test_manhattan_matrix_shape():
    assert manhattan_distance_matrix(WEIGHTS, A).shape == (5, 5)


def test_manhattan_matrix_matches_scalar():
    d = manhattan_distance_matrix(WEIGHTS, A)
    assert d[0, 4] == pytest.approx(manhattan_distance(WEIGHTS[0, 4], A))


def test_manhattan_geq_euclidean():
    # L1 >= L2 for same vectors
    assert manhattan_distance(A, B) >= euclidean_distance(A, B) - 1e-10


# ---------------------------------------------------------------------------
# Chebyshev
# ---------------------------------------------------------------------------

def test_chebyshev_identity():
    assert chebyshev_distance(A, A) == pytest.approx(0.0)


def test_chebyshev_non_negative():
    assert chebyshev_distance(A, B) >= 0.0


def test_chebyshev_known_value():
    a = np.array([1.0, 5.0, 2.0])
    b = np.array([4.0, 2.0, 3.0])
    assert chebyshev_distance(a, b) == pytest.approx(3.0)


def test_chebyshev_leq_manhattan():
    # L∞ <= L1 for any two vectors
    assert chebyshev_distance(A, B) <= manhattan_distance(A, B) + 1e-10


def test_chebyshev_matrix_shape():
    assert chebyshev_distance_matrix(WEIGHTS, A).shape == (5, 5)


def test_chebyshev_matrix_matches_scalar():
    d = chebyshev_distance_matrix(WEIGHTS, A)
    assert d[3, 1] == pytest.approx(chebyshev_distance(WEIGHTS[3, 1], A))


# ---------------------------------------------------------------------------
# Cosine
# ---------------------------------------------------------------------------

def test_cosine_identity():
    assert cosine_distance(A, A) == pytest.approx(0.0, abs=1e-10)


def test_cosine_range():
    d = cosine_distance(A, B)
    assert 0.0 <= d <= 1.0 + 1e-10


def test_cosine_zero_vector():
    assert cosine_distance(ZERO, A) == pytest.approx(1.0)


def test_cosine_antiparallel():
    # cosine distance = 1 - similarity; similarity of antiparallel = -1 → distance = 2
    assert cosine_distance(A, -A) == pytest.approx(2.0, abs=1e-10)


def test_cosine_matrix_shape():
    assert cosine_distance_matrix(WEIGHTS, A).shape == (5, 5)


def test_cosine_matrix_all_in_range():
    d = cosine_distance_matrix(WEIGHTS, A)
    # cosine distance ∈ [0, 2] (similarity ∈ [-1, 1])
    assert np.all(d >= -1e-10) and np.all(d <= 2.0 + 1e-10)


def test_cosine_matrix_matches_scalar():
    d = cosine_distance_matrix(WEIGHTS, A)
    assert d[2, 2] == pytest.approx(cosine_distance(WEIGHTS[2, 2], A), abs=1e-10)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def test_correlation_identity():
    assert correlation_distance(A, A) == pytest.approx(0.0, abs=1e-10)


def test_correlation_range():
    d = correlation_distance(A, B)
    assert 0.0 <= d <= 1.0 + 1e-10


def test_correlation_amplitude_invariant():
    # Scaling a signal should not change correlation distance
    assert correlation_distance(A, B) == pytest.approx(
        correlation_distance(3.0 * A, B), abs=1e-10
    )


def test_correlation_mean_invariant():
    # Adding a constant should not change correlation distance
    assert correlation_distance(A, B) == pytest.approx(
        correlation_distance(A + 100.0, B), abs=1e-10
    )


def test_correlation_zero_vector():
    assert correlation_distance(ZERO, A) == pytest.approx(1.0)


def test_correlation_matrix_shape():
    assert correlation_distance_matrix(WEIGHTS, A).shape == (5, 5)


def test_correlation_matrix_matches_scalar():
    d = correlation_distance_matrix(WEIGHTS, A)
    assert d[4, 0] == pytest.approx(correlation_distance(WEIGHTS[4, 0], A), abs=1e-10)


# ---------------------------------------------------------------------------
# DTW
# ---------------------------------------------------------------------------

def test_dtw_identity():
    assert dtw_distance(A, A) == pytest.approx(0.0, abs=1e-10)


def test_dtw_non_negative():
    assert dtw_distance(A, B) >= 0.0


def test_dtw_constant_sequences():
    a = np.ones(10) * 3.0
    b = np.ones(10) * 3.0
    assert dtw_distance(a, b) == pytest.approx(0.0)


def test_dtw_known_value():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert dtw_distance(a, b) == pytest.approx(0.0)


def test_dtw_with_band_non_negative():
    assert dtw_distance(A, B, band=5) >= 0.0


def test_dtw_band_geq_unconstrained():
    # Restricting the search band can only increase (or equal) the DTW cost
    d_free = dtw_distance(A, B)
    d_band = dtw_distance(A, B, band=3)
    assert d_band >= d_free - 1e-10


def test_dtw_matrix_shape():
    assert dtw_distance_matrix(WEIGHTS, A).shape == (5, 5)


def test_dtw_matrix_matches_scalar():
    d = dtw_distance_matrix(WEIGHTS, A)
    assert d[1, 3] == pytest.approx(dtw_distance(WEIGHTS[1, 3], A), abs=1e-10)


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def test_cross_correlation_identity():
    a_n = A / np.linalg.norm(A)
    # unit-norm signal: peak xcorr with itself = 1, so distance = 0
    assert cross_correlation_distance(a_n, a_n) == pytest.approx(0.0, abs=1e-10)


def test_cross_correlation_range():
    d = cross_correlation_distance(A, B)
    assert -1e-10 <= d <= 1.0 + 1e-10


def test_cross_correlation_zero_vector():
    assert cross_correlation_distance(ZERO, A) == pytest.approx(1.0)


def test_cross_correlation_matrix_shape():
    assert cross_correlation_distance_matrix(WEIGHTS, A).shape == (5, 5)


def test_cross_correlation_matrix_matches_scalar():
    d = cross_correlation_distance_matrix(WEIGHTS, A)
    assert d[0, 0] == pytest.approx(
        cross_correlation_distance(WEIGHTS[0, 0], A), abs=1e-10
    )


# ---------------------------------------------------------------------------
# Grid distances
# ---------------------------------------------------------------------------

@pytest.fixture
def ids_matrix():
    k = 4
    m = np.array([[[r, c] for c in range(k)] for r in range(k)])
    return m


def test_grid_euclidean_bmu_is_zero(ids_matrix):
    bmu = (2, 3)
    d = grid_euclidean(ids_matrix, bmu)
    assert d[bmu] == pytest.approx(0.0)


def test_grid_euclidean_shape(ids_matrix):
    assert grid_euclidean(ids_matrix, (0, 0)).shape == (4, 4)


def test_grid_euclidean_adjacent_equals_one(ids_matrix):
    d = grid_euclidean(ids_matrix, (2, 2))
    assert d[2, 3] == pytest.approx(1.0)
    assert d[3, 2] == pytest.approx(1.0)


def test_grid_euclidean_diagonal(ids_matrix):
    d = grid_euclidean(ids_matrix, (0, 0))
    assert d[1, 1] == pytest.approx(np.sqrt(2))


def test_grid_chebyshev_bmu_is_zero(ids_matrix):
    bmu = (1, 1)
    d = grid_chebyshev(ids_matrix, bmu)
    assert d[bmu] == pytest.approx(0.0)


def test_grid_chebyshev_adjacent_and_diagonal_both_one(ids_matrix):
    # Chebyshev: all 8 neighbours have distance 1
    d = grid_chebyshev(ids_matrix, (2, 2))
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        assert d[2 + dr, 2 + dc] == pytest.approx(1.0)
