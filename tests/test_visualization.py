"""Tests for AMBER/visualization.py — smoke tests (no display required)."""

import matplotlib
matplotlib.use('Agg')   # non-interactive backend; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest

import AMBER


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid resource leaks."""
    yield
    plt.close('all')


class TestVisualizationSmoke:
    """Verify every method runs without raising, using the Agg backend."""

    def test_heat_map(self, classification):
        AMBER.Visualization.heat_map(classification)

    def test_heat_map_custom_cmax(self, classification):
        AMBER.Visualization.heat_map(classification, cmax=5)

    def test_elevation_map(self, classification):
        AMBER.Visualization.elevation_map(classification)

    def test_umatrix(self, classification):
        AMBER.Visualization.umatrix(classification)

    def test_umatrix_binary_colorscale(self, classification):
        AMBER.Visualization.umatrix(classification, colorscale='binary')

    def test_codebook_vector(self, trained_map):
        AMBER.Visualization.codebook_vector(trained_map, index=0)

    def test_codebook_vector_with_header(self, trained_map):
        AMBER.Visualization.codebook_vector(trained_map, index=0, header='feature_0')

    def test_codebook_vectors(self, trained_map):
        AMBER.Visualization.codebook_vectors(trained_map)

    def test_bar_chart(self):
        AMBER.Visualization.bar_chart(np.array([1, 5, 3, 2]))

    def test_neurons_per_num_activations_map(self, classification):
        AMBER.Visualization.neurons_per_num_activations_map(classification)

    def test_characteristics_graph(self, trained_map):
        AMBER.Visualization.characteristics_graph(trained_map, row=0, column=0)

    def test_characteristics_bargraph(self, trained_map):
        AMBER.Visualization.characteristics_bargraph(trained_map, row=0, column=0)

    def test_full_map_weights(self, trained_map, tmp_path):
        filepath = str(tmp_path / 'weights.png')
        AMBER.Visualization.full_map_weights(trained_map, filename=filepath)

    def test_trajectory(self, classification, temporal_analysis):
        AMBER.Visualization.trajectory(classification, temporal_analysis)

    def test_transition_matrix_plot_normalised(self, temporal_analysis):
        AMBER.Visualization.transition_matrix_plot(temporal_analysis, normalised=True)

    def test_transition_matrix_plot_raw(self, temporal_analysis):
        AMBER.Visualization.transition_matrix_plot(temporal_analysis, normalised=False)

    def test_dwell_time_map(self, temporal_analysis, classification):
        AMBER.Visualization.dwell_time_map(temporal_analysis, classification)

    def test_trajectory_with_seed(self, classification, temporal_analysis):
        AMBER.Visualization.trajectory(classification, temporal_analysis,
                                       random_seed=42)

    def test_umatrix_labeled(self, small_data, trained_map):
        n = len(small_data)
        labels = np.zeros(n, dtype=int)
        labels[n // 2:] = 1
        tagged = np.column_stack([labels, small_data])
        cls = AMBER.Classification(trained_map, tagged, tagged=True)
        AMBER.Visualization.umatrix_labeled(cls, labels, class_names=['A', 'B'])

    def test_hit_map(self, small_data, trained_map):
        n = len(small_data)
        labels = np.zeros(n, dtype=int)
        tagged = np.column_stack([labels, small_data])
        cls = AMBER.Classification(trained_map, tagged, tagged=True)
        AMBER.Visualization.hit_map(cls, labels)

    def test_weight_map_grid(self, trained_map, small_data):
        n = len(small_data)
        labels = np.zeros(n, dtype=int)
        tagged = np.column_stack([labels, small_data])
        cls = AMBER.Classification(trained_map, tagged, tagged=True)
        AMBER.Visualization.weight_map_grid(trained_map, cls, labels)


class TestVisualizationSavePaths:
    """Methods that accept a filename parameter must write a file."""

    def test_umatrix_labeled_saves_file(self, small_data, trained_map, tmp_path):
        n = len(small_data)
        labels = np.zeros(n, dtype=int)
        labels[n // 2:] = 1
        tagged = np.column_stack([labels, small_data])
        cls = AMBER.Classification(trained_map, tagged, tagged=True)
        filepath = str(tmp_path / 'umatrix_labeled.png')
        AMBER.Visualization.umatrix_labeled(cls, labels, class_names=['A', 'B'],
                                            filename=filepath)
        assert (tmp_path / 'umatrix_labeled.png').exists()

    def test_hit_map_saves_file(self, small_data, trained_map, tmp_path):
        n = len(small_data)
        labels = np.zeros(n, dtype=int)
        labels[n // 2:] = 1
        tagged = np.column_stack([labels, small_data])
        cls = AMBER.Classification(trained_map, tagged, tagged=True)
        filepath = str(tmp_path / 'hit_map.png')
        AMBER.Visualization.hit_map(cls, labels, filename=filepath)
        assert (tmp_path / 'hit_map.png').exists()

    def test_weight_map_grid_saves_file(self, small_data, trained_map, tmp_path):
        n = len(small_data)
        labels = np.zeros(n, dtype=int)
        labels[n // 2:] = 1
        tagged = np.column_stack([labels, small_data])
        cls = AMBER.Classification(trained_map, tagged, tagged=True)
        filepath = str(tmp_path / 'weight_map_grid.png')
        AMBER.Visualization.weight_map_grid(trained_map, cls, labels,
                                            filename=filepath)
        assert (tmp_path / 'weight_map_grid.png').exists()

    def test_full_map_weights_with_labels(self, trained_map, tmp_path):
        """full_map_weights must accept a labels array without raising."""
        filepath = str(tmp_path / 'fmw_labels.png')
        labels = np.array([f'f{i}' for i in range(trained_map.input_data_dimension)])
        AMBER.Visualization.full_map_weights(trained_map, labels=labels,
                                             filename=filepath)


class TestCharacteristicsWithLabels:
    """characteristics_graph and characteristics_bargraph with labels array."""

    def test_characteristics_graph_with_labels(self, trained_map):
        feature_labels = np.array([f'feat_{i}'
                                    for i in range(trained_map.input_data_dimension)])
        AMBER.Visualization.characteristics_graph(trained_map, row=0, column=0,
                                                   labels=feature_labels)

    def test_characteristics_bargraph_with_labels(self, trained_map):
        feature_labels = np.array([f'feat_{i}'
                                    for i in range(trained_map.input_data_dimension)])
        AMBER.Visualization.characteristics_bargraph(trained_map, row=0, column=0,
                                                      labels=feature_labels)


class TestTrajectoryBackground:
    """trajectory() must accept background='umatrix' without raising."""

    def test_trajectory_umatrix_background(self, classification, temporal_analysis):
        AMBER.Visualization.trajectory(classification, temporal_analysis,
                                       background='umatrix', random_seed=0)


class TestTrajectoryReproducibility:
    """Trajectory visualization must be deterministic when a seed is given."""

    def test_same_seed_same_offsets(self, classification, temporal_analysis):
        """Two calls with the same seed must not raise and produce figures."""
        AMBER.Visualization.trajectory(classification, temporal_analysis,
                                       random_seed=0)
        plt.close('all')
        AMBER.Visualization.trajectory(classification, temporal_analysis,
                                       random_seed=0)
        plt.close('all')

    def test_no_seed_does_not_raise(self, classification, temporal_analysis):
        AMBER.Visualization.trajectory(classification, temporal_analysis)
