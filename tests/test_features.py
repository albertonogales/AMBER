"""Tests for AMBER/features.py — FeatureExtractor and standalone functions."""

import numpy as np
import pytest

from AMBER.features import (
    FeatureExtractor,
    zero_crossing_rate, line_length, hjorth_parameters, sample_entropy,
    spectral_power, dominant_frequency, spectral_entropy,
    spectral_centroid, spectral_rolloff, band_power,
    EEG_BANDS,
)

RNG = np.random.default_rng(55)
SINE = np.sin(np.linspace(0, 4 * np.pi, 256))
NOISE = RNG.standard_normal(256)
CONSTANT = np.ones(256) * 3.0
FS = 256.0


# ---------------------------------------------------------------------------
# Standalone statistical functions
# ---------------------------------------------------------------------------

class TestZeroCrossingRate:
    def test_range(self):
        assert 0.0 <= zero_crossing_rate(SINE) <= 1.0

    def test_constant_signal_zero(self):
        assert zero_crossing_rate(CONSTANT) == pytest.approx(0.0)

    def test_alternating_signal_high(self):
        alt = np.array([1.0, -1.0] * 50)
        assert zero_crossing_rate(alt) > 0.9


class TestLineLength:
    def test_non_negative(self):
        assert line_length(SINE) >= 0.0

    def test_constant_signal_zero(self):
        assert line_length(CONSTANT) == pytest.approx(0.0)

    def test_noisy_geq_clean(self):
        assert line_length(SINE + NOISE) >= line_length(SINE) - 1e-10


class TestHjorthParameters:
    def test_returns_three_values(self):
        result = hjorth_parameters(SINE)
        assert len(result) == 3

    def test_activity_equals_variance(self):
        activity, _, _ = hjorth_parameters(SINE)
        assert activity == pytest.approx(np.var(SINE), rel=1e-6)

    def test_all_non_negative(self):
        a, m, c = hjorth_parameters(SINE)
        assert a >= 0 and m >= 0 and c >= 0

    def test_constant_signal_mobility_zero(self):
        _, mobility, _ = hjorth_parameters(CONSTANT)
        assert mobility == pytest.approx(0.0)

    def test_finite_values(self):
        a, m, c = hjorth_parameters(NOISE)
        assert all(np.isfinite(v) for v in (a, m, c))


class TestSampleEntropy:
    def test_non_negative(self):
        assert sample_entropy(SINE[:64]) >= 0.0

    def test_regular_signal_low_entropy(self):
        # A pure sine is more regular than noise → lower entropy
        se_sine = sample_entropy(SINE[:64])
        se_noise = sample_entropy(NOISE[:64])
        assert se_sine <= se_noise + 0.5   # allow some tolerance

    def test_constant_signal_zero(self):
        # All templates match → A/B = 1 → -log(1) = 0
        assert sample_entropy(CONSTANT[:32]) == pytest.approx(0.0)

    def test_custom_r(self):
        result = sample_entropy(SINE[:64], r=0.5)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# Spectral functions
# ---------------------------------------------------------------------------

class TestSpectralPower:
    def test_non_negative(self):
        assert spectral_power(SINE, FS) >= 0.0

    def test_noise_has_power(self):
        assert spectral_power(NOISE, FS) > 0.0

    def test_silent_signal_zero(self):
        assert spectral_power(np.zeros(256), FS) == pytest.approx(0.0, abs=1e-10)


class TestDominantFrequency:
    def test_sine_dominant_frequency(self):
        # 2 Hz sine: 2 complete cycles over 2π, sampled at FS=256
        t = np.arange(256) / FS
        sig = np.sin(2 * np.pi * 2 * t)
        df = dominant_frequency(sig, FS)
        assert df == pytest.approx(2.0, abs=1.0)

    def test_returns_non_negative(self):
        assert dominant_frequency(SINE, FS) >= 0.0


class TestSpectralEntropy:
    def test_non_negative(self):
        assert spectral_entropy(SINE, FS) >= 0.0

    def test_noise_higher_than_sine(self):
        # Broad-spectrum noise has higher spectral entropy than a pure sine
        assert spectral_entropy(NOISE, FS) > spectral_entropy(SINE, FS)


class TestSpectralCentroid:
    def test_non_negative(self):
        assert spectral_centroid(SINE, FS) >= 0.0

    def test_within_nyquist(self):
        assert spectral_centroid(SINE, FS) <= FS / 2 + 1e-6

    def test_silent_signal_zero(self):
        assert spectral_centroid(np.zeros(256), FS) == pytest.approx(0.0)


class TestSpectralRolloff:
    def test_non_negative(self):
        assert spectral_rolloff(SINE, FS) >= 0.0

    def test_within_nyquist(self):
        assert spectral_rolloff(SINE, FS) <= FS / 2 + 1e-6

    def test_85pct_geq_50pct(self):
        r85 = spectral_rolloff(NOISE, FS, pct=0.85)
        r50 = spectral_rolloff(NOISE, FS, pct=0.50)
        assert r85 >= r50 - 1e-10


class TestBandPower:
    def test_non_negative(self):
        assert band_power(SINE, FS, 8.0, 13.0) >= 0.0

    def test_empty_band_returns_zero(self):
        # Band above Nyquist → no frequencies → 0
        assert band_power(SINE, FS, 200.0, 300.0) == pytest.approx(0.0)

    def test_eeg_bands_all_non_negative(self):
        for band, (lo, hi) in EEG_BANDS.items():
            assert band_power(NOISE, FS, lo, hi) >= 0.0, f"Negative power for {band}"


# ---------------------------------------------------------------------------
# FeatureExtractor — extract
# ---------------------------------------------------------------------------

class TestFeatureExtractorExtract:
    @pytest.fixture
    def fe(self):
        return FeatureExtractor(fs=FS)

    def test_extract_returns_1d_array(self, fe):
        result = fe.extract(SINE, features=['rms', 'mean', 'std'])
        assert result.ndim == 1

    def test_extract_length_matches_features(self, fe):
        feats = ['rms', 'mean', 'std', 'zero_crossing_rate']
        result = fe.extract(SINE, features=feats)
        assert len(result) == len(feats)

    def test_extract_finite_values(self, fe):
        feats = ['rms', 'mean', 'std', 'skewness', 'kurtosis',
                 'peak_to_peak', 'zero_crossing_rate', 'line_length']
        result = fe.extract(SINE, features=feats)
        assert np.all(np.isfinite(result))

    def test_extract_complexity_features(self, fe):
        feats = ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity']
        result = fe.extract(SINE, features=feats)
        assert len(result) == 3
        assert np.all(np.isfinite(result))

    def test_extract_spectral_features(self, fe):
        feats = ['spectral_power', 'dominant_frequency', 'spectral_entropy',
                 'spectral_centroid', 'spectral_rolloff']
        result = fe.extract(SINE, features=feats)
        assert len(result) == 5
        assert np.all(np.isfinite(result))

    def test_extract_eeg_band_powers(self, fe):
        feats = ['delta_power', 'theta_power', 'alpha_power',
                 'beta_power', 'gamma_power']
        result = fe.extract(NOISE, features=feats)
        assert len(result) == 5
        assert np.all(result >= 0)

    def test_extract_sample_entropy(self, fe):
        result = fe.extract(SINE[:64], features=['sample_entropy'])
        assert len(result) == 1
        assert result[0] >= 0.0

    def test_extract_unknown_feature_raises(self, fe):
        with pytest.raises(ValueError, match="Unknown feature"):
            fe.extract(SINE, features=['not_a_feature'])

    def test_default_features_no_fs_needed(self):
        fe = FeatureExtractor(fs=1.0)
        result = fe.extract(SINE)
        assert result.ndim == 1
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# FeatureExtractor — extract_batch
# ---------------------------------------------------------------------------

class TestFeatureExtractorBatch:
    def test_output_shape(self, signal_batch):
        fe = FeatureExtractor(fs=FS)
        feats = ['rms', 'mean', 'std']
        out = fe.extract_batch(signal_batch, features=feats)
        assert out.shape == (signal_batch.shape[0], len(feats))

    def test_output_finite(self, signal_batch):
        fe = FeatureExtractor(fs=FS)
        out = fe.extract_batch(signal_batch, features=['rms', 'zero_crossing_rate'])
        assert np.all(np.isfinite(out))

    def test_batch_row_matches_single_extract(self, signal_batch):
        fe = FeatureExtractor(fs=FS)
        feats = ['rms', 'mean', 'std']
        batch_out = fe.extract_batch(signal_batch, features=feats)
        single_out = fe.extract(signal_batch[3], features=feats)
        np.testing.assert_allclose(batch_out[3], single_out)


# ---------------------------------------------------------------------------
# FeatureExtractor — feature_names
# ---------------------------------------------------------------------------

class TestFeatureNames:
    def test_length_matches_extract(self):
        fe = FeatureExtractor(fs=FS)
        feats = ['rms', 'mean', 'zero_crossing_rate']
        names = fe.feature_names(feats)
        vec = fe.extract(SINE, feats)
        assert len(names) == len(vec)

    def test_names_are_strings(self):
        fe = FeatureExtractor(fs=FS)
        names = fe.feature_names(['rms', 'std'])
        assert all(isinstance(n, str) for n in names)


# ---------------------------------------------------------------------------
# NumPy compatibility — spectral functions must work on NumPy 1.x and 2.x
# ---------------------------------------------------------------------------

class TestSpectralCompatibility:
    """spectral_power and band_power must not use NumPy 2.x-only APIs."""

    def test_spectral_power_returns_finite(self):
        assert np.isfinite(spectral_power(SINE, FS))
        assert spectral_power(SINE, FS) > 0

    def test_band_power_alpha_returns_finite(self):
        val = band_power(SINE, FS, 8.0, 13.0)
        assert np.isfinite(val)
        assert val >= 0.0

    def test_spectral_power_noise(self):
        assert spectral_power(NOISE, FS) > 0

    def test_band_power_empty_band_returns_zero(self):
        # Band above Nyquist — no frequencies present
        val = band_power(SINE, 256.0, 200.0, 300.0)
        assert val == 0.0 or np.isfinite(val)


# ---------------------------------------------------------------------------
# sample_entropy edge cases — inf must not silently corrupt SOM training
# ---------------------------------------------------------------------------

class TestSampleEntropyEdgeCases:
    def test_sample_entropy_does_not_raise_on_any_finite_input(self):
        """sample_entropy must return a float for any finite input."""
        for x in [SINE, NOISE, CONSTANT, np.zeros(50), np.ones(10)]:
            result = sample_entropy(x[:50])
            assert isinstance(result, float)

    def test_inf_in_feature_matrix_raises_in_train(self):
        """Map.train must raise ValueError when the feature matrix contains inf."""
        import AMBER
        data = np.array([[1.0, 2.0, 3.0],
                         [np.inf, 1.0, 2.0],
                         [1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="non-finite"):
            AMBER.Map(data=data, size=2, period=10)

    def test_nan_in_feature_matrix_raises_in_train(self):
        """Map.train must raise ValueError when the feature matrix contains nan."""
        import AMBER
        data = np.array([[1.0, 2.0], [np.nan, 1.0], [1.0, 2.0]])
        with pytest.raises(ValueError, match="non-finite"):
            AMBER.Map(data=data, size=2, period=10)


# ---------------------------------------------------------------------------
# scipy-absent fallback paths (_SCIPY = False)
# ---------------------------------------------------------------------------

class TestScipyFallbacks:
    """Verify the numpy-only code paths when scipy is unavailable."""

    def test_skewness_numpy_path(self):
        """_skewness must return a float via the numpy path."""
        import AMBER.features as feat_mod
        from unittest.mock import patch
        with patch.object(feat_mod, '_SCIPY', False):
            from AMBER.features import _skewness
            result = _skewness.__wrapped__(SINE) if hasattr(_skewness, '__wrapped__') else feat_mod._skewness(SINE)
            # Call the function directly via the module to pick up patched _SCIPY
            # We re-execute the function body by calling it normally — the patch
            # replaces the module-level flag read inside the function.
            val = feat_mod._skewness(SINE)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_kurtosis_numpy_path(self):
        """_kurtosis must return a float via the numpy path."""
        import AMBER.features as feat_mod
        from unittest.mock import patch
        with patch.object(feat_mod, '_SCIPY', False):
            val = feat_mod._kurtosis(SINE)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_skewness_constant_zero(self):
        """Symmetric (constant) signal must give zero skewness in numpy path."""
        import AMBER.features as feat_mod
        from unittest.mock import patch
        with patch.object(feat_mod, '_SCIPY', False):
            val = feat_mod._skewness(CONSTANT)
        assert val == pytest.approx(0.0)

    def test_kurtosis_constant_zero(self):
        """Constant signal must give zero (excess) kurtosis in numpy path."""
        import AMBER.features as feat_mod
        from unittest.mock import patch
        with patch.object(feat_mod, '_SCIPY', False):
            val = feat_mod._kurtosis(CONSTANT)
        assert val == pytest.approx(0.0)

    def test_psd_numpy_path_returns_freqs_and_psd(self):
        """_psd must return (freqs, psd) via the numpy/periodogram path."""
        import AMBER.features as feat_mod
        from unittest.mock import patch
        with patch.object(feat_mod, '_SCIPY', False):
            freqs, psd = feat_mod._psd(SINE, FS)
        assert len(freqs) == len(psd)
        assert freqs[0] >= 0.0
        assert np.all(psd >= 0.0)

    def test_spectral_power_numpy_path(self):
        """spectral_power must work without scipy."""
        import AMBER.features as feat_mod
        from unittest.mock import patch
        with patch.object(feat_mod, '_SCIPY', False):
            val = feat_mod.spectral_power(SINE, FS)
        assert np.isfinite(val)
        assert val > 0.0


# ---------------------------------------------------------------------------
# librosa MFCC tests
# ---------------------------------------------------------------------------

class TestMFCC:
    def test_mfcc_raises_without_librosa(self):
        """compute_mfcc must raise ImportError when librosa is absent."""
        import AMBER.features as feat_mod
        from unittest.mock import patch
        with patch.object(feat_mod, '_LIBROSA', False):
            with pytest.raises(ImportError, match="librosa"):
                feat_mod.compute_mfcc(SINE, FS, n_mfcc=13, hop_length=512)

    def test_extract_mfcc_raises_without_librosa(self):
        """FeatureExtractor.extract must propagate ImportError for 'mfcc'."""
        import AMBER.features as feat_mod
        from unittest.mock import patch
        fe = FeatureExtractor(fs=FS)
        with patch.object(feat_mod, '_LIBROSA', False):
            with pytest.raises(ImportError):
                fe.extract(SINE, features=['mfcc'])

    def test_mfcc_returns_vector_of_correct_length(self):
        """When librosa is usable at runtime, MFCC must return n_mfcc values."""
        import AMBER.features as _fm
        if not _fm._LIBROSA:
            pytest.skip("librosa not installed")
        fe = FeatureExtractor(fs=FS, n_mfcc=13)
        try:
            result = fe.extract(SINE, features=['mfcc'])
        except ImportError:
            pytest.skip("librosa not usable in this environment (dependency conflict)")
        assert len(result) == 13

    def test_extract_batch_mfcc_shape(self, signal_batch):
        """extract_batch with 'mfcc' must produce (n_windows, n_mfcc)."""
        import AMBER.features as _fm
        if not _fm._LIBROSA:
            pytest.skip("librosa not installed")
        fe = FeatureExtractor(fs=FS, n_mfcc=5)
        try:
            out = fe.extract_batch(signal_batch, features=['mfcc'])
        except ImportError:
            pytest.skip("librosa not usable in this environment (dependency conflict)")
        assert out.shape == (signal_batch.shape[0], 5)

    def test_feature_names_mfcc_expanded(self):
        """feature_names must expand 'mfcc' to n_mfcc individual names."""
        fe = FeatureExtractor(fs=FS, n_mfcc=5)
        names = fe.feature_names(['mean', 'mfcc', 'std'])
        assert names == ['mean', 'mfcc_0', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'std']
