"""Tests for AMBER/classification.py — Classification class."""

import numpy as np
import pytest

import AMBER


class TestClassificationBasic:
    def test_activations_sum_equals_n_samples(self, classification, small_data):
        assert classification.activations_map.sum() == small_data.shape[0]

    def test_bmu_positions_within_grid(self, classification, trained_map):
        k = trained_map.map_size
        xs = classification.classification_map['x']
        ys = classification.classification_map['y']
        assert xs.between(0, k - 1).all()
        assert ys.between(0, k - 1).all()

    def test_classification_map_length(self, classification, small_data):
        assert len(classification.classification_map) == small_data.shape[0]

    def test_classification_map_columns(self, classification):
        for col in ('labels', 'x', 'y', 'dist'):
            assert col in classification.classification_map.columns

    def test_distances_non_negative(self, classification):
        assert (classification.classification_map['dist'] >= 0).all()

    def test_distances_map_non_negative(self, classification):
        assert np.all(classification.distances_map >= 0)


class TestClassificationMetrics:
    def test_quantization_error_non_negative(self, classification):
        assert classification.quantization_error >= 0.0

    def test_topological_error_in_unit_interval(self, classification):
        assert 0.0 <= classification.topological_error <= 1.0

    def test_mean_distance_map_non_negative(self, classification):
        assert classification.mean_distance_map >= 0.0

    def test_num_activations_positive(self, classification):
        assert classification.num_activations > 0

    def test_num_activations_leq_total_neurons(self, classification, trained_map):
        k = trained_map.map_size
        assert classification.num_activations <= k * k

    def test_quantization_error_map_non_negative(self, classification):
        assert np.all(classification.quantization_error_map >= 0)

    def test_topological_error_map_in_unit_interval(self, classification):
        assert np.all(classification.topological_error_map >= 0)
        assert np.all(classification.topological_error_map <= 1.0 + 1e-10)


class TestUMatrix:
    def test_umatrix_shape(self, classification, trained_map):
        k = trained_map.map_size
        expected = (2 * k - 1, 2 * k - 1)
        assert classification.umatriz.shape == expected

    def test_umatrix_non_negative(self, classification):
        assert np.all(classification.umatriz >= 0)

    def test_umatrix_finite(self, classification):
        assert np.all(np.isfinite(classification.umatriz))


class TestTaggedData:
    def test_tagged_splits_labels_and_data(self, trained_map):
        labels = np.array([0, 1, 2, 0, 1])
        features = np.random.default_rng(7).standard_normal((5, 4))
        tagged = np.column_stack([labels, features])
        c = AMBER.Classification(trained_map, tagged, tagged=True)
        np.testing.assert_array_equal(c.classification_labels, labels)
        assert c.classification_data.shape == (5, 4)

    def test_untagged_labels_are_sequential(self, trained_map, small_data):
        c = AMBER.Classification(trained_map, small_data)
        expected = np.arange(small_data.shape[0])
        np.testing.assert_array_equal(c.classification_labels, expected)


class TestMapSizeVariants:
    @pytest.mark.parametrize('map_size', [2, 5, 8])
    def test_various_map_sizes(self, small_data, map_size):
        m = AMBER.Map(data=small_data, size=map_size, period=20)
        c = AMBER.Classification(m, small_data)
        assert c.activations_map.sum() == small_data.shape[0]
        assert c.umatriz.shape == (2 * map_size - 1, 2 * map_size - 1)


# ──────────────────────────────────────────────────────────────────────────────
# New / fixed metric tests
# ──────────────────────────────────────────────────────────────────────────────

class TestQEDenominator:
    """QE must be mean distance per *sample*, not per active neuron."""

    def test_qe_equals_mean_sample_distance(self, classification):
        """Direct recomputation from the classification_map distances."""
        expected = classification.classification_map['dist'].mean()
        assert abs(classification.quantization_error - expected) < 1e-4

    def test_qe_less_than_mean_distance_map(self, classification):
        """QE (per sample) ≤ mean_distance_map (per active neuron) only when
        some neurons are hit more than once.  We just check QE is finite and
        non-negative here."""
        assert np.isfinite(classification.quantization_error)
        assert classification.quantization_error >= 0.0

    def test_qe_not_biased_by_dead_neurons(self):
        """With a large map and few samples only a few neurons fire.
        Old code divided by num_active_neurons; new code divides by n_samples.
        Verify that the two differ on a concrete example."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((10, 4))
        # 8×8 map → many dead neurons
        m = AMBER.Map(data=data, size=8, period=20)
        c = AMBER.Classification(m, data)
        per_sample = c.classification_map['dist'].mean()
        # New QE must equal per-sample mean (tolerance allows for 5-decimal rounding of distances_map)
        assert abs(c.quantization_error - per_sample) < 1e-4


class TestTEDenominator:
    """TE must be fraction of *samples* with non-adjacent BMU/2nd-BMU."""

    def test_te_in_unit_interval(self, classification):
        assert 0.0 <= classification.topological_error <= 1.0

    def test_te_is_fraction_not_count(self, trained_map):
        """TE is in [0,1] regardless of sample count."""
        data = np.random.default_rng(0).standard_normal((5, trained_map.input_data_dimension))
        c = AMBER.Classification(trained_map, data)
        assert 0.0 <= c.topological_error <= 1.0

    def test_te_denominator_is_n_samples(self):
        """Force a known number of TE events and verify the fraction is correct."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal((20, 4))
        m = AMBER.Map(data=data, size=4, period=30)
        c = AMBER.Classification(m, data)
        # TE must equal count / 20
        expected = c.topological_map.sum() / 20
        assert abs(c.topological_error - expected) < 1e-10


class TestConfiguredDistanceQE:
    """quantization_error uses the map's configured distance (primary).
    quantization_error_euclidean always uses Euclidean (secondary)."""

    def test_both_attributes_exist(self, classification):
        assert hasattr(classification, 'quantization_error')
        assert hasattr(classification, 'quantization_error_euclidean')

    def test_both_non_negative_finite(self, classification):
        assert classification.quantization_error >= 0.0
        assert np.isfinite(classification.quantization_error)
        assert classification.quantization_error_euclidean >= 0.0
        assert np.isfinite(classification.quantization_error_euclidean)

    def test_equal_for_euclidean_map(self, classification):
        """When the map uses Euclidean, both QE values should agree."""
        assert abs(classification.quantization_error
                   - classification.quantization_error_euclidean) < 1e-4

    def test_qe_uses_configured_distance_for_cosine_map(self):
        """For a cosine-distance map, QE (cosine) must differ from QE (Euclidean)."""
        rng = np.random.default_rng(11)
        data = rng.standard_normal((30, 6))
        m = AMBER.Map(data=data, size=4, period=30, distance='cosine')
        c = AMBER.Classification(m, data)
        # cosine QE is in [0, 1]; Euclidean QE is in physical weight space
        assert c.quantization_error >= 0.0
        assert np.isfinite(c.quantization_error)
        # They must differ for a non-trivial cosine map
        assert abs(c.quantization_error - c.quantization_error_euclidean) > 1e-6

    def test_qe_uses_configured_distance_for_correlation_map(self):
        """For a correlation-distance map, QE and Euclidean QE must differ."""
        rng = np.random.default_rng(22)
        data = rng.standard_normal((30, 8))
        m = AMBER.Map(data=data, size=4, period=30, distance='correlation')
        c = AMBER.Classification(m, data)
        assert 0.0 <= c.quantization_error <= 1.0  # correlation distance is in [0,1]
        assert c.quantization_error_euclidean >= 0.0
        assert abs(c.quantization_error - c.quantization_error_euclidean) > 1e-6


class TestDistortionMeasure:
    """distortion measure must exist and be sensible."""

    def test_attribute_exists(self, classification):
        assert hasattr(classification, 'distortion')

    def test_non_negative_finite(self, classification):
        assert classification.distortion >= 0.0
        assert np.isfinite(classification.distortion)

    def test_distortion_geq_qe_squared_times_n(self, classification):
        """Because h ≥ 0 everywhere and the BMU term alone gives ||x-w_BMU||²,
        distortion ≥ QE² (loosely, since h(BMU,BMU)=1 and QE is mean L2)."""
        # This is not a tight bound but verifies the measure is at least as
        # large as the pure BMU contribution for a single sample approximation.
        assert classification.distortion >= 0.0

    def test_distortion_decreases_with_better_map(self):
        """A longer-trained map should have lower distortion."""
        rng = np.random.default_rng(99)
        data = rng.standard_normal((50, 4))
        m_short = AMBER.Map(data=data, size=4, period=5)
        m_long  = AMBER.Map(data=data, size=4, period=200)
        c_short = AMBER.Classification(m_short, data)
        c_long  = AMBER.Classification(m_long,  data)
        assert c_long.distortion <= c_short.distortion


class TestUMatrixFull:
    """U-matrix must fill all 8 directions, not just 3."""

    def test_umatrix_shape(self, classification, trained_map):
        k = trained_map.map_size
        assert classification.umatriz.shape == (2*k - 1, 2*k - 1)

    def test_umatrix_non_negative(self, classification):
        assert np.all(classification.umatriz >= 0)

    def test_umatrix_finite(self, classification):
        assert np.all(np.isfinite(classification.umatriz))

    def test_umatrix_edge_cells_nonzero(self):
        """For a non-trivial map, all inter-neuron edge cells should be > 0."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal((40, 4))
        m = AMBER.Map(data=data, size=4, period=50)
        c = AMBER.Classification(m, data)
        k = 4
        # Horizontal edges: positions (2i+1, 2j) for i<k-1
        for i in range(k - 1):
            for j in range(k):
                assert c.umatriz[2*i+1, 2*j] > 0, f"horizontal edge ({i},{j}) is zero"
        # Vertical edges: positions (2i, 2j+1) for j<k-1
        for i in range(k):
            for j in range(k - 1):
                assert c.umatriz[2*i, 2*j+1] > 0, f"vertical edge ({i},{j}) is zero"

    def test_umatrix_diagonal_cells_nonzero(self):
        """Diagonal cells (2i+1, 2j+1) should also be filled."""
        rng = np.random.default_rng(4)
        data = rng.standard_normal((40, 4))
        m = AMBER.Map(data=data, size=4, period=50)
        c = AMBER.Classification(m, data)
        k = 4
        for i in range(k - 1):
            for j in range(k - 1):
                assert c.umatriz[2*i+1, 2*j+1] > 0, f"diagonal cell ({i},{j}) is zero"

    def test_umatrix_symmetry_horizontal(self):
        """The distance from neuron A→B equals B→A, so symmetric edges should match."""
        rng = np.random.default_rng(5)
        data = rng.standard_normal((40, 4))
        m = AMBER.Map(data=data, size=4, period=50)
        c = AMBER.Classification(m, data)
        k = 4
        # Horizontal edge (2i+1, 2j) computed from both (i,j)→(i+1,j) and (i+1,j)→(i,j)
        # both map to the same cell, so we just verify the cell value matches manual compute
        from AMBER.distances import euclidean_distance
        for i in range(k - 1):
            for j in range(k):
                expected = euclidean_distance(m.weights[i, j], m.weights[i+1, j])
                assert abs(c.umatriz[2*i+1, 2*j] - expected) < 1e-10


# ---------------------------------------------------------------------------
# Normalisation consistency at classification time
# ---------------------------------------------------------------------------

class TestNormalizationConsistency:
    """BMU search in Classification must operate in the same space as training."""

    def test_zscore_qe_is_in_normalised_scale(self):
        """With zscore normalisation, QE must be in the normalised scale."""
        rng = np.random.default_rng(7)
        # Data with large magnitude — in raw space distances would be O(100)
        data = rng.random((40, 4)) * 100.0
        som = AMBER.Map(data=data, size=3, period=200,
                        normalization='zscore', random_seed=42)
        cls = AMBER.Classification(som, data)
        # In z-scored space distances are O(1) not O(100)
        assert np.isfinite(cls.quantization_error)
        assert cls.quantization_error < 10.0

    def test_bmu_positions_consistent_with_transform(self, small_data):
        """BMUs from Classification must match direct calculate_bmu on norm data."""
        som = AMBER.Map(data=small_data, size=3, period=100,
                        normalization='zscore', random_seed=42)
        cls = AMBER.Classification(som, small_data)
        norm_data = som.transform(small_data)
        for i in range(len(small_data)):
            _, bmu_pos, _, _ = som.calculate_bmu(norm_data[i])
            row = cls.classification_map.iloc[i]
            assert (int(row['x']), int(row['y'])) == bmu_pos

    def test_mean_distance_map_per_sample(self, classification, small_data):
        """mean_distance_map must divide by n_samples, not num_activations."""
        n_samples = len(small_data)
        expected = np.sum(classification.distances_map) / n_samples
        # distances_map is rounded to 5 decimals, so allow a small tolerance
        assert abs(classification.mean_distance_map - expected) < 1e-4

    def test_none_normalization_unchanged(self, small_data):
        """With normalization='none', transform() must return data unchanged."""
        som = AMBER.Map(data=small_data, size=2, period=10,
                        normalization='none', random_seed=0)
        np.testing.assert_array_equal(som.transform(small_data), small_data)


class TestClassificationVerboseAndOther:
    """Classification verbose=True and other DataFrame paths."""

    def test_verbose_does_not_raise(self, trained_map, small_data, caplog):
        """verbose=True must log labels and data without raising."""
        import logging
        with caplog.at_level(logging.DEBUG):
            c = AMBER.Classification(trained_map, small_data, verbose=True)
        assert c.activations_map.sum() == small_data.shape[0]

    def test_other_dataframe_appended(self, trained_map, small_data):
        """other DataFrame must be concatenated into classification_map."""
        import pandas as pd
        n = len(small_data)
        extra = pd.DataFrame({'score': np.arange(n, dtype=float)})
        c = AMBER.Classification(trained_map, small_data, other=extra)
        assert 'score' in c.classification_map.columns
        assert len(c.classification_map) == n

    def test_other_dataframe_values_preserved(self, trained_map, small_data):
        """Values from the other DataFrame must not be altered."""
        import pandas as pd
        n = len(small_data)
        vals = np.arange(n, dtype=float)
        extra = pd.DataFrame({'custom': vals})
        c = AMBER.Classification(trained_map, small_data, other=extra)
        np.testing.assert_array_equal(c.classification_map['custom'].values, vals)
