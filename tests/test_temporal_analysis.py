"""Tests for AMBER/temporal_analysis.py — TemporalAnalysis."""

import numpy as np
import pytest

import AMBER


class TestTransitionMatrix:
    def test_shape(self, temporal_analysis, trained_map):
        k = trained_map.map_size
        expected = (k * k, k * k)
        assert temporal_analysis.transition_matrix.shape == expected

    def test_non_negative(self, temporal_analysis):
        assert np.all(temporal_analysis.transition_matrix >= 0)

    def test_total_transitions_equals_n_minus_one(self, temporal_analysis, small_data):
        assert temporal_analysis.transition_matrix.sum() == small_data.shape[0] - 1

    def test_normalised_row_sums(self, temporal_analysis):
        T = temporal_analysis.transition_matrix_norm
        row_sums = T.sum(axis=1)
        # Rows with outgoing transitions should sum to 1
        active_rows = temporal_analysis.transition_matrix.sum(axis=1) > 0
        np.testing.assert_allclose(row_sums[active_rows], 1.0, atol=1e-10)

    def test_normalised_rows_non_negative(self, temporal_analysis):
        assert np.all(temporal_analysis.transition_matrix_norm >= 0)


class TestStability:
    def test_range(self, temporal_analysis):
        assert 0.0 <= temporal_analysis.stability <= 1.0

    def test_perfect_stability(self, trained_map, small_data):
        # Classify the same pattern repeatedly → BMU never changes → stability = 1
        same_data = np.tile(small_data[0], (10, 1))
        c = AMBER.Classification(trained_map, same_data)
        ta = AMBER.TemporalAnalysis(c)
        assert ta.stability == pytest.approx(1.0)

    def test_single_sample_stability_one(self, trained_map, small_data):
        c = AMBER.Classification(trained_map, small_data[:1])
        ta = AMBER.TemporalAnalysis(c)
        assert ta.stability == pytest.approx(1.0)


class TestMeanPathLength:
    def test_non_negative(self, temporal_analysis):
        assert temporal_analysis.mean_path_length >= 0.0

    def test_stationary_path_is_zero(self, trained_map, small_data):
        same_data = np.tile(small_data[0], (10, 1))
        c = AMBER.Classification(trained_map, same_data)
        ta = AMBER.TemporalAnalysis(c)
        assert ta.mean_path_length == pytest.approx(0.0)

    def test_finite(self, temporal_analysis):
        assert np.isfinite(temporal_analysis.mean_path_length)


class TestTrajectory:
    def test_length_equals_n_samples(self, temporal_analysis, small_data):
        assert len(temporal_analysis.trajectory) == small_data.shape[0]

    def test_positions_within_grid(self, temporal_analysis, trained_map):
        k = trained_map.map_size
        for row, col in temporal_analysis.trajectory:
            assert 0 <= row < k
            assert 0 <= col < k

    def test_trajectory_is_list_of_tuples(self, temporal_analysis):
        assert all(
            isinstance(p, tuple) and len(p) == 2
            for p in temporal_analysis.trajectory
        )


class TestDwellTimes:
    def test_all_values_geq_one(self, temporal_analysis):
        for pos, mean_d in temporal_analysis.dwell_times().items():
            assert mean_d >= 1.0, f"Dwell time < 1 at {pos}"

    def test_constant_sequence_dwell_equals_length(self, trained_map, small_data):
        same_data = np.tile(small_data[0], (8, 1))
        c = AMBER.Classification(trained_map, same_data)
        ta = AMBER.TemporalAnalysis(c)
        dwell = ta.dwell_times()
        assert len(dwell) == 1
        assert list(dwell.values())[0] == pytest.approx(8.0)


class TestMostFrequentTransitions:
    def test_returns_list(self, temporal_analysis):
        result = temporal_analysis.most_frequent_transitions(5)
        assert isinstance(result, list)

    def test_count_sorted_descending(self, temporal_analysis):
        result = temporal_analysis.most_frequent_transitions(10)
        counts = [t['count'] for t in result]
        assert counts == sorted(counts, reverse=True)

    def test_each_entry_has_required_keys(self, temporal_analysis):
        for entry in temporal_analysis.most_frequent_transitions(3):
            assert 'from' in entry and 'to' in entry and 'count' in entry

    def test_top_k_limit(self, temporal_analysis):
        result = temporal_analysis.most_frequent_transitions(3)
        assert len(result) <= 3


class TestSummary:
    def test_summary_runs_without_error(self, temporal_analysis, capsys):
        temporal_analysis.summary()
        captured = capsys.readouterr()
        assert 'Stability' in captured.out
        assert 'Unique BMUs' in captured.out
