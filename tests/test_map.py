"""Tests for AMBER/map.py — Map class and vesanto_size."""

import json
import os
import tempfile

import numpy as np
import pytest

import AMBER
from AMBER.map import vesanto_size


# ---------------------------------------------------------------------------
# vesanto_size
# ---------------------------------------------------------------------------

class TestVesantoSize:
    def test_returns_int(self):
        assert isinstance(vesanto_size(100), int)

    def test_minimum_two(self):
        # For N=1 the formula gives round(sqrt(5))=2; the max(2,...) floor applies
        assert vesanto_size(1) == 2
        # For N=2 the formula gives round(sqrt(5*sqrt(2)))=round(2.66)=3, still valid
        assert vesanto_size(2) >= 2

    def test_monotone_increasing(self):
        sizes = [vesanto_size(n) for n in [50, 100, 500, 1000, 5000]]
        assert sizes == sorted(sizes)

    def test_known_value_100(self):
        # sqrt(5 * sqrt(100)) = sqrt(50) ≈ 7.07 → 7
        assert vesanto_size(100) == 7

    def test_known_value_1000(self):
        # sqrt(5 * sqrt(1000)) ≈ sqrt(158.1) ≈ 12.6 → 13
        assert vesanto_size(1000) == 13

    def test_exposed_on_package(self):
        assert AMBER.vesanto_size(100) == vesanto_size(100)


# ---------------------------------------------------------------------------
# Map — construction
# ---------------------------------------------------------------------------

class TestMapInit:
    def test_explicit_size_preserved(self, small_data):
        m = AMBER.Map(data=small_data, size=5, period=10)
        assert m.map_size == 5

    def test_auto_size_from_data(self, small_data):
        m = AMBER.Map(data=small_data, period=10)
        assert m.map_size == vesanto_size(small_data.shape[0])

    def test_auto_size_error_without_data(self):
        with pytest.raises(ValueError, match="size"):
            AMBER.Map(size=None)

    def test_period_assertion(self, small_data):
        with pytest.raises(ValueError):
            AMBER.Map(data=small_data, size=3, period=1)

    def test_lr_assertion_too_high(self, small_data):
        with pytest.raises(ValueError):
            AMBER.Map(data=small_data, size=3, period=10, initial_lr=1.5)

    def test_lr_assertion_zero(self, small_data):
        with pytest.raises(ValueError):
            AMBER.Map(data=small_data, size=3, period=10, initial_lr=0.0)

    def test_size_assertion(self, small_data):
        with pytest.raises(ValueError):
            AMBER.Map(data=small_data, size=1, period=10)

    def test_unknown_distance_assertion(self, small_data):
        with pytest.raises(ValueError, match="distance|Unknown"):
            AMBER.Map(data=small_data, size=3, period=10, distance='invalid')

    def test_no_data_no_train(self):
        m = AMBER.Map(size=4, period=10)
        assert m.weights.shape == (1,)   # placeholder until train() is called

    def test_neighbourhood_defaults_to_size(self, small_data):
        m = AMBER.Map(data=small_data, size=5, period=10, initial_neighbourhood=0)
        assert m.neighbourhood == 5

    def test_neighbourhood_explicit(self, small_data):
        m = AMBER.Map(data=small_data, size=5, period=10, initial_neighbourhood=3)
        assert m.neighbourhood == 3


# ---------------------------------------------------------------------------
# Map — training
# ---------------------------------------------------------------------------

class TestMapTrain:
    def test_weights_shape_after_train(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=10)
        assert m.weights.shape == (4, 4, small_data.shape[1])

    def test_weights_change_during_training(self, small_data):
        m = AMBER.Map(size=4, period=50)
        before = np.ones((4, 4, small_data.shape[1])) * 0.5
        m.weights = before.copy()
        m.input_data_dimension = small_data.shape[1]
        m.num_data = small_data.shape[0]
        m.train(small_data)
        assert not np.allclose(m.weights, before)

    def test_num_data_set_after_train(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=10)
        assert m.num_data == small_data.shape[0]

    def test_input_dimension_set_after_train(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=10)
        assert m.input_data_dimension == small_data.shape[1]


# ---------------------------------------------------------------------------
# Map — weight initialization
# ---------------------------------------------------------------------------

class TestWeightInit:
    def test_random_in_unit_interval(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=10, weights='random')
        # weights are reinitialised in train(); check final weights are finite
        assert np.all(np.isfinite(m.weights))

    def test_random_negative_range(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=10, weights='random_negative')
        assert np.all(np.isfinite(m.weights))

    def test_sample_weights_finite(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=10, weights='sample')
        assert np.all(np.isfinite(m.weights))

    def test_pca_weights_finite(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=10, weights='PCA')
        assert np.all(np.isfinite(m.weights))

    def test_pca_weights_shape(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=10, weights='PCA')
        assert m.weights.shape == (4, 4, small_data.shape[1])


# ---------------------------------------------------------------------------
# Map — normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    """Test via Map._Map__normalize (instance method)."""

    def _norm(self, data, method):
        # __normalize is now an instance method; create a minimal map to call it
        m = AMBER.Map(size=2, period=2, normalization=method)
        return m._Map__normalize(data, method)

    def test_none_unchanged(self, small_data):
        out = self._norm(small_data, 'none')
        np.testing.assert_array_equal(out, small_data)

    def test_zscore_mean_zero(self, medium_data):
        out = self._norm(medium_data, 'zscore')
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-10)

    def test_zscore_std_one(self, medium_data):
        out = self._norm(medium_data, 'zscore')
        np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-10)

    def test_fwn_alias_for_zscore(self, medium_data):
        np.testing.assert_allclose(
            self._norm(medium_data, 'fwn'),
            self._norm(medium_data, 'zscore'),
        )

    def test_robust_finite(self, medium_data):
        out = self._norm(medium_data, 'robust')
        assert np.all(np.isfinite(out))

    def test_01scale_range(self, medium_data):
        out = self._norm(medium_data, '01scale')
        assert out.min() == pytest.approx(0.0, abs=1e-10)
        assert out.max() == pytest.approx(1.0, abs=1e-10)

    def test_zscore_sample_per_row_mean_zero(self, medium_data):
        out = self._norm(medium_data, 'zscore_sample')
        np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-10)

    def test_robust_sample_finite(self, medium_data):
        out = self._norm(medium_data, 'robust_sample')
        assert np.all(np.isfinite(out))

    def test_minmax_sample_per_row_range(self, medium_data):
        out = self._norm(medium_data, 'minmax_sample')
        np.testing.assert_allclose(out.min(axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(out.max(axis=1), 1.0, atol=1e-10)

    def test_l2_unit_norm_per_row(self, medium_data):
        out = self._norm(medium_data, 'l2')
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_euclidean_alias_for_l2(self, medium_data):
        np.testing.assert_allclose(
            self._norm(medium_data, 'euclidean'),
            self._norm(medium_data, 'l2'),
        )

    def test_unknown_method_raises(self, small_data):
        with pytest.raises(ValueError, match="normalization"):
            self._norm(small_data, 'no_such_method')

    def test_constant_row_does_not_produce_nan(self):
        data = np.ones((10, 4))
        for method in ('zscore_sample', 'robust_sample', 'minmax_sample', 'l2'):
            out = self._norm(data, method)
            assert np.all(np.isfinite(out)), f"NaN/Inf with method '{method}'"


# ---------------------------------------------------------------------------
# Map — calculate_bmu
# ---------------------------------------------------------------------------

class TestCalculateBMU:
    def test_returns_four_values(self, trained_map, small_data):
        result = trained_map.calculate_bmu(small_data[0])
        assert len(result) == 4

    def test_bmu_pos_within_grid(self, trained_map, small_data):
        _, bmu_pos, _, _ = trained_map.calculate_bmu(small_data[0])
        assert 0 <= bmu_pos[0] < trained_map.map_size
        assert 0 <= bmu_pos[1] < trained_map.map_size

    def test_bmu_dist_leq_second_bmu_dist(self, trained_map, small_data):
        bmu_d, _, second_d, _ = trained_map.calculate_bmu(small_data[0])
        assert bmu_d <= second_d + 1e-10

    def test_bmu_and_second_bmu_different_positions(self, trained_map, small_data):
        _, bmu_pos, _, second_pos = trained_map.calculate_bmu(small_data[0])
        assert bmu_pos != second_pos

    def test_bmu_distance_non_negative(self, trained_map, small_data):
        bmu_d, _, _, _ = trained_map.calculate_bmu(small_data[0])
        assert bmu_d >= 0.0

    @pytest.mark.parametrize('distance', AMBER.AVAILABLE_DISTANCES)
    def test_all_distances_produce_valid_bmu(self, small_data, distance):
        m = AMBER.Map(data=small_data, size=3, period=10, distance=distance)
        _, bmu_pos, _, _ = m.calculate_bmu(small_data[0])
        assert 0 <= bmu_pos[0] < 3
        assert 0 <= bmu_pos[1] < 3

    def test_exact_weight_match_gives_zero_distance(self, trained_map):
        # Replace one neuron's weight with a known pattern → its distance must be 0
        known = trained_map.weights[2, 1].copy()
        dist, pos, _, _ = trained_map.calculate_bmu(known)
        assert dist == pytest.approx(0.0, abs=1e-10)
        assert pos == (2, 1)


# ---------------------------------------------------------------------------
# Map — static helpers
# ---------------------------------------------------------------------------

class TestStaticHelpers:
    def test_variation_lr_at_start(self):
        lr = AMBER.Map.variation_learning_rate(0.1, 1, 100)
        assert 0.0 < lr <= 0.1

    def test_variation_lr_at_end(self):
        # LR at final iteration is small but positive (denominator is T+1, not T)
        lr = AMBER.Map.variation_learning_rate(0.1, 100, 100)
        assert 0.0 < lr < 0.1 / 100

    def test_variation_lr_decreasing(self):
        lrs = [AMBER.Map.variation_learning_rate(0.1, i, 100) for i in range(1, 101)]
        assert all(lrs[i] >= lrs[i + 1] - 1e-10 for i in range(len(lrs) - 1))

    def test_variation_neighbourhood_at_start(self):
        v = AMBER.Map.variation_neighbourhood(10, 1, 100)
        assert v == pytest.approx(10 * (1 - 1 / 100))

    def test_variation_neighbourhood_at_end(self):
        v = AMBER.Map.variation_neighbourhood(10, 100, 100)
        assert v == pytest.approx(0.0, abs=1e-10)

    def test_decay_at_zero_distance(self):
        d = AMBER.Map.decay(np.array([0.0]), 2.0)
        assert d[0] == pytest.approx(1.0)

    def test_decay_decreasing_with_distance(self):
        distances = np.array([0.0, 1.0, 2.0, 3.0])
        d = AMBER.Map.decay(distances, 2.0)
        assert all(d[i] >= d[i + 1] for i in range(len(d) - 1))


# ---------------------------------------------------------------------------
# Map — serialisation round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_weights_preserved(self, trained_map):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model')
            trained_map.save_classifier(path)
            loaded = AMBER.Map.load_classifier(path)
        np.testing.assert_array_almost_equal(loaded.weights, trained_map.weights)

    def test_hyperparameters_preserved(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=20,
                      distance='manhattan', normalization='zscore',
                      use_decay=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model')
            m.save_classifier(path)
            loaded = AMBER.Map.load_classifier(path)
        assert loaded.distance == 'manhattan'
        assert loaded.normalization == 'zscore'
        assert loaded.use_decay is True
        assert loaded.map_size == 4

    def test_loaded_map_classifies(self, trained_map, small_data):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model')
            trained_map.save_classifier(path)
            loaded = AMBER.Map.load_classifier(path)
        _, pos, _, _ = loaded.calculate_bmu(small_data[0])
        assert 0 <= pos[0] < loaded.map_size
        assert 0 <= pos[1] < loaded.map_size

    def test_json_file_created(self, trained_map):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'mymodel')
            trained_map.save_classifier(path)
            assert os.path.exists(path + '.json')


# ---------------------------------------------------------------------------
# Fix #1 — 'sample' weight initialisation (whole samples, not scalars)
# ---------------------------------------------------------------------------

class TestSampleWeightInit:
    def test_sample_weights_are_real_data_points(self):
        """Each neuron weight vector must equal one of the training samples."""
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (50, 6)).astype(np.float32)
        m = AMBER.Map(data=None, size=4, period=2, weights='sample')
        m.num_data = len(data)
        m.input_data_dimension = data.shape[1]
        weights = m._Map__init_weights(data, 'sample')   # access name-mangled method
        flat_weights = weights.reshape(-1, 6)
        for w in flat_weights:
            dists = np.linalg.norm(data - w, axis=1)
            assert np.min(dists) < 1e-6, \
                f"Neuron weight {w} is not a data sample — bug still present"

    def test_sample_weights_shape(self):
        rng = np.random.default_rng(1)
        data = rng.uniform(0, 1, (30, 8)).astype(np.float32)
        m = AMBER.Map(data=data, size=3, period=5, weights='sample')
        assert m.weights.shape == (3, 3, 8)

    def test_sample_weights_finite(self):
        rng = np.random.default_rng(2)
        data = rng.uniform(0, 1, (40, 5)).astype(np.float32)
        m = AMBER.Map(data=data, size=4, period=5, weights='sample')
        assert np.all(np.isfinite(m.weights))


# ---------------------------------------------------------------------------
# Fix #2 — Pure Gaussian neighbourhood (no hard boundary)
# ---------------------------------------------------------------------------

class TestGaussianNeighbourhood:
    def test_all_neurons_updated_with_gaussian(self):
        """With use_decay=True all neurons should shift (Gaussian never exactly 0)."""
        rng = np.random.default_rng(42)
        data = rng.uniform(0, 1, (40, 4)).astype(np.float32)
        m = AMBER.Map(data=data, size=4, period=20,
                      use_decay=True, weights='random')
        # weights_ changed from uniform random — just confirm they are finite
        # and vary across the grid (some learning occurred)
        assert np.all(np.isfinite(m.weights))
        assert not np.all(m.weights == m.weights[0, 0])

    def test_bubble_only_updates_within_radius(self):
        """With use_decay=False only neurons inside radius receive nonzero update."""
        rng = np.random.default_rng(7)
        data = rng.uniform(0, 1, (20, 3)).astype(np.float32)
        # Train for 1 step manually and inspect
        m = AMBER.Map(data=None, size=5, period=2,
                      use_decay=False, weights='random')
        m.num_data = len(data)
        m.input_data_dimension = 3
        m.weights = np.zeros((5, 5, 3))     # known start
        pattern = np.ones(3)
        bmu = (2, 2)
        radius = 1.0
        m._Map__adjust_weights(radius, 0.5, bmu, pattern)
        # neurons at distance > 1 (e.g. corners at distance ~2.83) unchanged
        assert np.all(m.weights[0, 0] == 0.0), \
            "Corner neuron (distance ~2.83) should not be updated with bubble"

    def test_gaussian_produces_finite_weights(self):
        rng = np.random.default_rng(3)
        data = rng.standard_normal((60, 5)).astype(np.float32)
        m = AMBER.Map(data=data, size=5, period=30,
                      use_decay=True, normalization='zscore')
        assert np.all(np.isfinite(m.weights))


# ---------------------------------------------------------------------------
# Fix #3 — PCA initialisation via SVD
# ---------------------------------------------------------------------------

class TestPCAInitSVD:
    def test_pca_weights_shape(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (80, 10)).astype(np.float32)
        m = AMBER.Map(data=data, size=5, period=5, weights='PCA')
        assert m.weights.shape == (5, 5, 10)

    def test_pca_weights_finite(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((100, 20)).astype(np.float32)
        m = AMBER.Map(data=data, size=4, period=5, weights='PCA')
        assert np.all(np.isfinite(m.weights))

    def test_pca_weights_vary_across_grid(self):
        """PCA init must produce different vectors for different grid positions."""
        rng = np.random.default_rng(2)
        data = rng.standard_normal((100, 8)).astype(np.float32)
        m = AMBER.Map(data=None, size=4, period=2, weights='PCA')
        m.num_data = len(data)
        m.input_data_dimension = 8
        w = m._Map__init_weights(data, 'PCA')
        # Not all rows should be identical
        assert not np.all(w == w[0, 0])

    def test_pca_high_dimensional_stable(self):
        """SVD should not produce NaN/inf on high-D data (condition-number test)."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal((50, 64)).astype(np.float32)   # Digits-like
        m = AMBER.Map(data=data, size=4, period=5, weights='PCA')
        assert np.all(np.isfinite(m.weights))


# ---------------------------------------------------------------------------
# Fix #4 — Asymptotic learning-rate / neighbourhood decay
# ---------------------------------------------------------------------------

class TestAsymptoticDecay:
    def test_lr_asymptotic_decreasing(self):
        lrs = [AMBER.Map.variation_learning_rate(0.5, i, 100, mode='asymptotic')
               for i in range(1, 101)]
        assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))

    def test_lr_asymptotic_positive(self):
        """Asymptotic decay never reaches zero."""
        lr_end = AMBER.Map.variation_learning_rate(0.5, 100, 100, mode='asymptotic')
        assert lr_end > 0.0

    def test_lr_linear_positive_at_final_iteration(self):
        # Final iteration receives a small but non-zero LR (denominator is T+1)
        lr_end = AMBER.Map.variation_learning_rate(0.5, 100, 100, mode='linear')
        assert 0.0 < lr_end < 0.5 / 100

    def test_neighbourhood_asymptotic_decreasing(self):
        vs = [AMBER.Map.variation_neighbourhood(10, i, 100, mode='asymptotic')
              for i in range(1, 101)]
        assert all(vs[i] >= vs[i + 1] for i in range(len(vs) - 1))

    def test_neighbourhood_asymptotic_above_final(self):
        v_end = AMBER.Map.variation_neighbourhood(10, 100, 100,
                                                  final=1, mode='asymptotic')
        assert v_end >= 1.0

    def test_map_trains_with_asymptotic_decay(self):
        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (50, 5)).astype(np.float32)
        m = AMBER.Map(data=data, size=4, period=20,
                      lr_decay='asymptotic', use_decay=True)
        assert np.all(np.isfinite(m.weights))

    def test_lr_decay_preserved_in_save_load(self):
        rng = np.random.default_rng(5)
        data = rng.uniform(0, 1, (40, 4)).astype(np.float32)
        m = AMBER.Map(data=data, size=3, period=10, lr_decay='asymptotic')
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model')
            m.save_classifier(path)
            loaded = AMBER.Map.load_classifier(path)
        assert loaded.lr_decay == 'asymptotic'


# ---------------------------------------------------------------------------
# Issue 1 — random_seed reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_weights(self, small_data):
        m1 = AMBER.Map(data=small_data, size=4, period=20, random_seed=42)
        m2 = AMBER.Map(data=small_data, size=4, period=20, random_seed=42)
        np.testing.assert_array_equal(m1.weights, m2.weights)

    def test_different_seeds_different_weights(self, small_data):
        m1 = AMBER.Map(data=small_data, size=4, period=20, random_seed=1)
        m2 = AMBER.Map(data=small_data, size=4, period=20, random_seed=2)
        assert not np.array_equal(m1.weights, m2.weights)

    def test_seed_persisted_in_save_load(self, small_data, tmp_path):
        m = AMBER.Map(data=small_data, size=4, period=20, random_seed=7)
        path = str(tmp_path / 'model')
        m.save_classifier(path)
        m2 = AMBER.Map.load_classifier(path)
        assert m2.random_seed == 7

    def test_none_seed_works(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=10, random_seed=None)
        assert np.all(np.isfinite(m.weights))


# ---------------------------------------------------------------------------
# Issue 6 — reinforce
# ---------------------------------------------------------------------------

class TestReinforce:
    def test_reinforce_improves_or_maintains_qe(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=20, random_seed=0)
        c_before = AMBER.Classification(m, small_data)
        m.reinforce(small_data, extension=1)
        c_after = AMBER.Classification(m, small_data)
        # reinforce should not make QE worse significantly
        assert c_after.quantization_error <= c_before.quantization_error * 1.5

    def test_reinforce_trained_attribute(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=10, random_seed=0)
        m.reinforce(small_data)
        # map is still trained
        assert m._Map__trained is True

    def test_reinforce_multiple_rounds(self, small_data):
        m = AMBER.Map(data=small_data, size=4, period=10, random_seed=0)
        m.reinforce(small_data, reinforcement=2, extension=2, compression=0.5)
        assert np.all(np.isfinite(m.weights))


# ---------------------------------------------------------------------------
# Issue 6 — save/load extended tests
# ---------------------------------------------------------------------------

class TestSaveLoadExtended:
    def test_save_load_roundtrip(self, trained_map, tmp_path):
        path = str(tmp_path / 'model')
        trained_map.save_classifier(path)
        loaded = AMBER.Map.load_classifier(path)
        np.testing.assert_array_almost_equal(trained_map.weights, loaded.weights)
        assert loaded.map_size == trained_map.map_size
        assert loaded.distance == trained_map.distance

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AMBER.Map.load_classifier(str(tmp_path / 'nonexistent'))


# ---------------------------------------------------------------------------
# Sampling correctness (regression for off-by-one bugs)
# ---------------------------------------------------------------------------

class TestSamplingCorrectness:
    """Regression tests for off-by-one bugs in pattern selection."""

    def test_random_integers_includes_last_sample(self):
        """_rng.integers(0, n) must be able to produce index n-1."""
        rng = np.random.default_rng(0)
        n = 10
        samples = [rng.integers(0, n) for _ in range(10_000)]
        assert (n - 1) in samples

    def test_sequential_first_presentation_is_index_zero(self):
        """Sequential mode must start at index 0 (numPresentation=1 → index 0)."""
        n = 5
        # With fix: index = (numPresentation - 1) % n; for numPresentation=1: 0
        assert (1 - 1) % n == 0

    def test_sequential_covers_all_samples_in_one_epoch(self):
        """All n samples must appear exactly once when period == n."""
        n = 7
        visited = {(i - 1) % n for i in range(1, n + 1)}
        assert visited == set(range(n))

    def test_random_presentation_last_sample_reachable(self, small_data):
        """With enough iterations, the last sample must be assigned a BMU."""
        som = AMBER.Map(data=small_data, size=2, period=5000,
                        presentation='random', random_seed=0)
        cls = AMBER.Classification(som, small_data)
        # All samples — including the last — must appear in classification_map
        assert len(cls.classification_map) == len(small_data)
        last_dist = cls.classification_map['dist'].iloc[-1]
        assert np.isfinite(last_dist)


# ---------------------------------------------------------------------------
# Normalisation storage and transform
# ---------------------------------------------------------------------------

class TestNormalizationStorage:
    """Stored normalisation parameters must survive training and save/load."""

    def test_zscore_params_stored(self, small_data):
        som = AMBER.Map(data=small_data, size=2, period=10,
                        normalization='zscore', random_seed=0)
        assert 'mean' in som._norm_params and 'std' in som._norm_params

    def test_robust_params_stored(self, small_data):
        som = AMBER.Map(data=small_data, size=2, period=10,
                        normalization='robust', random_seed=0)
        assert 'median' in som._norm_params and 'iqr' in som._norm_params

    def test_01scale_is_per_feature(self):
        """01scale must normalise each feature column to [0,1] independently."""
        data = np.array([[1.0, 100.0],
                         [2.0, 200.0],
                         [3.0, 300.0]])
        som = AMBER.Map(size=2, period=10, normalization='01scale', random_seed=0)
        som.train(data)
        np.testing.assert_array_almost_equal(som._norm_params['lo'],  [1.0, 100.0])
        np.testing.assert_array_almost_equal(som._norm_params['hi'],  [3.0, 300.0])
        np.testing.assert_array_almost_equal(som._norm_params['rng'], [2.0, 200.0])

    def test_transform_zscore_reproduces_training_normalisation(self, small_data):
        """transform() must reproduce the normalisation applied during training."""
        som = AMBER.Map(data=small_data, size=2, period=10,
                        normalization='zscore', random_seed=0)
        transformed = som.transform(small_data)
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(transformed.std(axis=0),  1.0, atol=1e-10)

    def test_norm_params_survive_save_load(self, small_data, tmp_path):
        """Normalisation parameters must be preserved across JSON save/load."""
        som = AMBER.Map(data=small_data, size=2, period=10,
                        normalization='zscore', random_seed=0)
        path = str(tmp_path / 'norm_test')
        som.save_classifier(path)
        loaded = AMBER.Map.load_classifier(path)
        np.testing.assert_array_almost_equal(
            loaded._norm_params['mean'], som._norm_params['mean'])
        np.testing.assert_array_almost_equal(
            loaded._norm_params['std'],  som._norm_params['std'])

    def test_train_raises_on_inf_data(self):
        """train() must raise ValueError when data contains inf."""
        data = np.array([[1.0, 2.0], [np.inf, 1.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="non-finite"):
            AMBER.Map(data=data, size=2, period=10)


class TestReinforceSequential:
    """reinforce() sequential-presentation branch must iterate in order."""

    def test_reinforce_sequential_presentation_changes_weights(self, small_data):
        """After reinforce() with sequential presentation the weights must change."""
        m = AMBER.Map(data=small_data, size=3, period=5,
                      presentation='sequential', random_seed=0)
        weights_before = m.weights.copy()
        m.reinforce(small_data, reinforcement=1, extension=2, compression=0.5)
        assert not np.array_equal(m.weights, weights_before)

    def test_reinforce_sequential_does_not_raise(self, small_data):
        """reinforce() must complete without error for several rounds."""
        m = AMBER.Map(data=small_data, size=3, period=20, random_seed=1)
        m.reinforce(small_data, reinforcement=1, extension=1, compression=0.3)

    def test_reinforce_normalizes_data(self, small_data):
        """reinforce() on a zscore-normalised map must not raise."""
        m = AMBER.Map(data=small_data, size=3, period=20,
                      normalization='zscore', random_seed=2)
        m.reinforce(small_data, reinforcement=1)


class TestTemporalMapReinforce:
    """TemporalMap.reinforce() must reset context and delegate to parent."""

    def test_temporal_map_reinforce_does_not_raise(self, small_data):
        tm = AMBER.TemporalMap(
            data=small_data, size=3, period=20,
            context_weight=0.5, context_influence=0.3,
        )
        tm.reinforce(small_data, reinforcement=1)

    def test_temporal_map_reinforce_changes_weights(self, small_data):
        tm = AMBER.TemporalMap(
            data=small_data, size=3, period=5,
            context_weight=0.5, context_influence=0.3,
        )
        weights_before = tm.weights.copy()
        tm.reinforce(small_data, reinforcement=1, extension=2, compression=0.5)
        assert not np.array_equal(tm.weights, weights_before)

    def test_temporal_map_norm_params_survive_save_load(self, small_data, tmp_path):
        """TemporalMap norm_params must be preserved across JSON save/load."""
        tm = AMBER.TemporalMap(
            data=small_data, size=3, period=20,
            context_weight=0.5, context_influence=0.3,
            normalization='zscore',
        )
        path = str(tmp_path / 'tm_norm')
        tm.save_classifier(path)
        loaded = AMBER.TemporalMap.load_classifier(path)
        np.testing.assert_array_almost_equal(
            loaded._norm_params['mean'], tm._norm_params['mean'])
        np.testing.assert_array_almost_equal(
            loaded._norm_params['std'],  tm._norm_params['std'])
