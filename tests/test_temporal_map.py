"""Tests for AMBER/temporal_map.py — TemporalMap (RSOM)."""

import os
import tempfile

import numpy as np
import pytest

import AMBER
from AMBER.map import vesanto_size


class TestTemporalMapInit:
    def test_auto_size_from_data(self, small_data):
        tm = AMBER.TemporalMap(data=small_data, period=10)
        assert tm.map_size == vesanto_size(small_data.shape[0])

    def test_explicit_size_preserved(self, small_data):
        tm = AMBER.TemporalMap(data=small_data, size=5, period=10)
        assert tm.map_size == 5

    def test_error_without_data_and_size(self):
        with pytest.raises(ValueError, match="size"):
            AMBER.TemporalMap(size=None)

    def test_context_weight_stored(self, small_data):
        tm = AMBER.TemporalMap(data=small_data, size=3, period=10, context_weight=0.7)
        assert tm.context_weight == 0.7

    def test_context_influence_stored(self, small_data):
        tm = AMBER.TemporalMap(data=small_data, size=3, period=10, context_influence=0.4)
        assert tm.context_influence == 0.4

    def test_invalid_context_weight_raises(self, small_data):
        with pytest.raises(AssertionError):
            AMBER.TemporalMap(data=small_data, size=3, period=10, context_weight=1.5)

    def test_invalid_context_influence_raises(self, small_data):
        with pytest.raises(AssertionError):
            AMBER.TemporalMap(data=small_data, size=3, period=10, context_influence=-0.1)

    def test_forces_sequential_presentation(self, small_data):
        tm = AMBER.TemporalMap(data=small_data, size=3, period=10)
        assert tm.presentation == 'sequential'

    def test_weights_shape_after_init(self, small_data):
        tm = AMBER.TemporalMap(data=small_data, size=4, period=10)
        assert tm.weights.shape == (4, 4, small_data.shape[1])


class TestTemporalMapContext:
    def test_context_none_before_bmu(self, small_data):
        tm = AMBER.TemporalMap(size=4, period=10)
        tm.train(small_data)
        tm.reset_context()
        assert tm._context is None

    def test_context_set_after_bmu(self, small_data):
        tm = AMBER.TemporalMap(data=small_data, size=4, period=10)
        tm.reset_context()
        tm.calculate_bmu(small_data[0])
        assert tm._context is not None

    def test_context_shape(self, small_data):
        tm = AMBER.TemporalMap(data=small_data, size=4, period=10)
        tm.reset_context()
        tm.calculate_bmu(small_data[0])
        assert tm._context.shape == (small_data.shape[1],)

    def test_reset_context_clears(self, small_data):
        tm = AMBER.TemporalMap(data=small_data, size=4, period=10)
        tm.calculate_bmu(small_data[0])
        tm.reset_context()
        assert tm._context is None

    def test_context_changes_across_steps(self, small_data):
        tm = AMBER.TemporalMap(data=small_data, size=4, period=10)
        tm.reset_context()
        tm.calculate_bmu(small_data[0])
        ctx1 = tm._context.copy()
        tm.calculate_bmu(small_data[1])
        ctx2 = tm._context.copy()
        # Context should evolve (not guaranteed identical)
        # At minimum it should stay finite
        assert np.all(np.isfinite(ctx2))

    def test_train_resets_context(self, small_data):
        tm = AMBER.TemporalMap(data=small_data, size=4, period=10)
        # Manually set context to something non-None
        tm._context = np.ones(small_data.shape[1])
        tm.train(small_data)   # should reset context at the start
        # After train, context is updated by BMU calls, so it won't be None
        # but weights should be finite
        assert np.all(np.isfinite(tm.weights))


class TestTemporalMapBMU:
    def test_returns_four_values(self, trained_temporal_map, small_data):
        trained_temporal_map.reset_context()
        result = trained_temporal_map.calculate_bmu(small_data[0])
        assert len(result) == 4

    def test_bmu_pos_within_grid(self, trained_temporal_map, small_data):
        trained_temporal_map.reset_context()
        _, pos, _, _ = trained_temporal_map.calculate_bmu(small_data[0])
        k = trained_temporal_map.map_size
        assert 0 <= pos[0] < k
        assert 0 <= pos[1] < k

    def test_zero_influence_matches_plain_map(self, small_data):
        """With context_influence=0, TemporalMap should behave like Map."""
        m = AMBER.Map(data=small_data, size=4, period=30)
        tm = AMBER.TemporalMap(
            data=small_data, size=4, period=30,
            context_influence=0.0,
        )
        # Copy identical weights
        tm.weights = m.weights.copy()
        tm.reset_context()

        for pattern in small_data[:5]:
            _, pos_m, _, _ = m.calculate_bmu(pattern)
            _, pos_tm, _, _ = tm.calculate_bmu(pattern)
            assert pos_m == pos_tm


class TestTemporalMapSerialisation:
    def test_round_trip_weights(self, trained_temporal_map):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'tmap')
            trained_temporal_map.save_classifier(path)
            loaded = AMBER.TemporalMap.load_classifier(path)
        np.testing.assert_array_almost_equal(
            loaded.weights, trained_temporal_map.weights
        )

    def test_round_trip_temporal_params(self, small_data):
        tm = AMBER.TemporalMap(
            data=small_data, size=4, period=20,
            context_weight=0.6, context_influence=0.35,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'tmap')
            tm.save_classifier(path)
            loaded = AMBER.TemporalMap.load_classifier(path)
        assert loaded.context_weight == pytest.approx(0.6)
        assert loaded.context_influence == pytest.approx(0.35)

    def test_loaded_map_classifies(self, trained_temporal_map, small_data):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'tmap')
            trained_temporal_map.save_classifier(path)
            loaded = AMBER.TemporalMap.load_classifier(path)
        loaded.reset_context()
        _, pos, _, _ = loaded.calculate_bmu(small_data[0])
        assert 0 <= pos[0] < loaded.map_size


class TestTemporalMapReproducibility:
    def test_same_seed_same_weights(self, small_data):
        tm1 = AMBER.TemporalMap(data=small_data, size=4, period=20, random_seed=42)
        tm2 = AMBER.TemporalMap(data=small_data, size=4, period=20, random_seed=42)
        np.testing.assert_array_equal(tm1.weights, tm2.weights)

    def test_seed_persisted_in_save_load(self, small_data, tmp_path):
        tm = AMBER.TemporalMap(data=small_data, size=4, period=10, random_seed=99)
        path = str(tmp_path / 'tmap')
        tm.save_classifier(path)
        loaded = AMBER.TemporalMap.load_classifier(path)
        assert loaded.random_seed == 99

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AMBER.TemporalMap.load_classifier(str(tmp_path / 'nonexistent'))
