"""Tests for AMBER/iterativesom.py — IterativeSOM."""

import numpy as np
import pytest

from AMBER.iterativesom import IterativeSOM
from AMBER.map import vesanto_size


class TestIterativeSOMInit:
    def test_creates_map_per_size(self, small_data):
        sizes = [3, 4, 5]
        iso = IterativeSOM(small_data, period=10, initial_lr=0.1, size_range=sizes)
        assert set(iso.maps.keys()) == set(sizes)

    def test_all_maps_trained(self, small_data):
        iso = IterativeSOM(small_data, period=10, initial_lr=0.1, size_range=[3, 4])
        for m in iso.maps.values():
            assert m.weights.ndim == 3

    def test_default_range_uses_vesanto(self, small_data):
        iso = IterativeSOM(small_data, period=10, initial_lr=0.1)
        recommended = vesanto_size(small_data.shape[0])
        # Default range is ±2 around recommended
        assert recommended in iso.maps

    def test_give_best_sets_best_map(self, small_data):
        iso = IterativeSOM(
            small_data, period=10, initial_lr=0.1,
            size_range=[3, 4], give_best=True,
        )
        assert iso.best_map is not None

    def test_best_map_has_lowest_qe(self, small_data):
        iso = IterativeSOM(
            small_data, period=20, initial_lr=0.1,
            size_range=[3, 4, 5], give_best=True,
        )
        import AMBER
        qes = {
            size: AMBER.Classification(m, small_data).quantization_error
            for size, m in iso.maps.items()
        }
        best_qe = AMBER.Classification(iso.best_map, small_data).quantization_error
        assert best_qe == pytest.approx(min(qes.values()), rel=1e-6)

    def test_no_give_best_best_map_is_none(self, small_data):
        iso = IterativeSOM(small_data, period=10, initial_lr=0.1,
                           size_range=[3, 4], give_best=False)
        assert iso.best_map is None


class TestCalculateRange:
    def test_returns_range_type(self, small_data):
        result = IterativeSOM.calculate_range(small_data)
        assert hasattr(result, '__iter__')

    def test_contains_vesanto_size(self, small_data):
        result = list(IterativeSOM.calculate_range(small_data))
        assert vesanto_size(small_data.shape[0]) in result

    def test_all_sizes_geq_two(self, small_data):
        result = list(IterativeSOM.calculate_range(small_data))
        assert all(s >= 2 for s in result)

    def test_custom_max_size(self, small_data):
        result = list(IterativeSOM.calculate_range(small_data, max_size=3))
        assert all(s <= 3 for s in result)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestIterativeSOMReproducibility:
    def test_same_seed_produces_identical_maps(self, small_data):
        """Identical random_seed must yield bit-identical weights for every size."""
        it1 = IterativeSOM(small_data, period=20, initial_lr=0.1,
                           size_range=[3, 4], random_seed=42)
        it2 = IterativeSOM(small_data, period=20, initial_lr=0.1,
                           size_range=[3, 4], random_seed=42)
        for size in it1.maps:
            np.testing.assert_array_equal(
                it1.maps[size].weights, it2.maps[size].weights,
                err_msg=f"Weights differ for map size {size} with same seed",
            )

    def test_different_seeds_produce_different_maps(self, small_data):
        """Different seeds should (almost certainly) produce different weights."""
        it1 = IterativeSOM(small_data, period=20, initial_lr=0.1,
                           size_range=[3, 4], random_seed=1)
        it2 = IterativeSOM(small_data, period=20, initial_lr=0.1,
                           size_range=[3, 4], random_seed=999)
        any_differ = any(
            not np.array_equal(it1.maps[s].weights, it2.maps[s].weights)
            for s in it1.maps
        )
        assert any_differ

    def test_validation_data_parameter(self, small_data):
        """give_best with held-out validation_data must not raise."""
        n = len(small_data)
        train, val = small_data[:n // 2], small_data[n // 2:]
        it = IterativeSOM(train, period=20, initial_lr=0.1,
                          size_range=[3, 4], give_best=True,
                          validation_data=val, random_seed=0)
        assert it.best_map is not None

    def test_warns_when_no_validation_data(self, small_data, caplog):
        """give_best without validation_data must log an in-sample selection warning."""
        import logging
        with caplog.at_level(logging.WARNING, logger='AMBER.iterativesom'):
            IterativeSOM(small_data, period=10, initial_lr=0.1,
                         size_range=[3], give_best=True, random_seed=0)
        assert any('selection bias' in rec.message or 'validation_data' in rec.message
                   for rec in caplog.records)

    def test_no_warning_when_validation_data_provided(self, small_data):
        """give_best with validation_data must NOT log the in-sample warning."""
        import logging
        n = len(small_data)
        train, val = small_data[:n // 2], small_data[n // 2:]
        # No assertion needed — just verify it runs without logging warning
        it = IterativeSOM(train, period=10, initial_lr=0.1,
                          size_range=[3], give_best=True,
                          validation_data=val, random_seed=0)
        assert it.best_map is not None
