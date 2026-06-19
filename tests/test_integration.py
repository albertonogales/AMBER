"""End-to-end integration tests for the full AMBER workflow."""

import os
import tempfile

import numpy as np
import pytest

import AMBER


RNG = np.random.default_rng(77)


class TestFullWorkflow:
    """Map → Classification → TemporalAnalysis pipeline."""

    def test_complete_pipeline(self):
        data = RNG.standard_normal((60, 6))
        som = AMBER.Map(data=data, period=30)
        c = AMBER.Classification(som, data)
        ta = AMBER.TemporalAnalysis(c)

        assert c.activations_map.sum() == data.shape[0]
        assert 0.0 <= ta.stability <= 1.0
        assert ta.mean_path_length >= 0.0

    def test_feature_extraction_into_map(self, signal_batch):
        fe = AMBER.FeatureExtractor(fs=256)
        feats = ['rms', 'zero_crossing_rate', 'hjorth_activity',
                 'hjorth_mobility', 'spectral_entropy']
        X = fe.extract_batch(signal_batch, features=feats)

        assert X.shape == (signal_batch.shape[0], len(feats))

        som = AMBER.Map(data=X, period=30, normalization='zscore_sample')
        c = AMBER.Classification(som, X)

        assert c.activations_map.sum() == X.shape[0]
        assert c.quantization_error >= 0.0

    def test_save_load_classify_consistency(self):
        data = RNG.standard_normal((40, 5))
        som = AMBER.Map(data=data, size=4, period=30)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'model')
            som.save_classifier(path)
            loaded = AMBER.Map.load_classifier(path)

        for pattern in data[:5]:
            _, pos_orig, _, _ = som.calculate_bmu(pattern)
            _, pos_load, _, _ = loaded.calculate_bmu(pattern)
            assert pos_orig == pos_load

    def test_temporal_map_pipeline(self):
        data = RNG.standard_normal((50, 5))
        tsom = AMBER.TemporalMap(data=data, size=4, period=30,
                                 context_weight=0.5, context_influence=0.3)
        tsom.reset_context()
        c = AMBER.Classification(tsom, data)
        ta = AMBER.TemporalAnalysis(c)

        assert c.activations_map.sum() == data.shape[0]
        assert 0.0 <= ta.stability <= 1.0

    def test_iterative_som_best_map_classifies(self):
        data = RNG.standard_normal((60, 4))
        from AMBER.iterativesom import IterativeSOM
        iso = IterativeSOM(data, period=20, initial_lr=0.1,
                           size_range=[3, 4, 5], give_best=True)
        c = AMBER.Classification(iso.best_map, data)
        assert c.activations_map.sum() == data.shape[0]

    def test_all_distances_complete_pipeline(self):
        data = RNG.standard_normal((30, 8))
        for dist in AMBER.AVAILABLE_DISTANCES:
            som = AMBER.Map(data=data, size=3, period=10, distance=dist)
            c = AMBER.Classification(som, data)
            assert c.activations_map.sum() == data.shape[0], \
                f"Pipeline failed for distance '{dist}'"

    def test_all_normalizations_complete_pipeline(self):
        data = np.abs(RNG.standard_normal((40, 5))) + 0.1  # positive for stability
        methods = ['none', 'zscore', 'fwn', 'robust', '01scale',
                   'zscore_sample', 'robust_sample', 'minmax_sample', 'l2']
        for method in methods:
            som = AMBER.Map(data=data, size=3, period=10, normalization=method)
            c = AMBER.Classification(som, data)
            assert c.activations_map.sum() == data.shape[0], \
                f"Pipeline failed for normalization '{method}'"


class TestEdgeCases:
    def test_minimum_map_size(self):
        data = RNG.standard_normal((20, 3))
        som = AMBER.Map(data=data, size=2, period=10)
        c = AMBER.Classification(som, data)
        assert c.activations_map.sum() == data.shape[0]

    def test_single_feature(self):
        data = RNG.standard_normal((20, 1))
        som = AMBER.Map(data=data, size=3, period=10)
        c = AMBER.Classification(som, data)
        assert c.activations_map.sum() == data.shape[0]

    def test_single_sample_classification(self, trained_map, small_data):
        c = AMBER.Classification(trained_map, small_data[:1])
        assert c.activations_map.sum() == 1
        ta = AMBER.TemporalAnalysis(c)
        assert ta.stability == pytest.approx(1.0)

    def test_dtw_band_pipeline(self):
        data = RNG.standard_normal((20, 6))
        som = AMBER.Map(data=data, size=3, period=10,
                        distance='dtw', dtw_band=2)
        c = AMBER.Classification(som, data)
        assert c.activations_map.sum() == data.shape[0]

    def test_pca_weight_init_pipeline(self):
        data = RNG.standard_normal((40, 5))
        som = AMBER.Map(data=data, size=4, period=20, weights='PCA')
        c = AMBER.Classification(som, data)
        assert c.activations_map.sum() == data.shape[0]

    def test_use_decay_pipeline(self):
        data = RNG.standard_normal((30, 4))
        som = AMBER.Map(data=data, size=3, period=20, use_decay=True)
        c = AMBER.Classification(som, data)
        assert c.activations_map.sum() == data.shape[0]
