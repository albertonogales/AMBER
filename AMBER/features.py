"""
Feature extraction for time series (biosignals, audio).

FeatureExtractor converts raw signal windows into fixed-length vectors for Map.train/classify.
Requires librosa for 'mfcc'; scipy improves spectral accuracy but falls back to numpy.
"""

import numpy as np

# np.trapezoid introduced in NumPy 2.0; np.trapz removed in NumPy 2.0.
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

try:
    from scipy import signal as _sp_signal
    from scipy import stats as _sp_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import librosa as _librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False


EEG_BANDS = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0,  13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 100.0),
}


class FeatureExtractor:
    """Extracts a flat feature vector from a 1-D signal window.

    Usage::

        fe = FeatureExtractor(fs=256)
        x = fe.extract(signal, features=['rms', 'spectral_entropy', 'hjorth_activity'])
        X = fe.extract_batch(windows, features=['rms', 'zero_crossing_rate', 'alpha_power'])

    Available features
    ------------------
    Statistical (no fs): mean, std, var, skewness, kurtosis, rms, peak_to_peak,
        zero_crossing_rate, line_length
    Spectral (fs required): spectral_power, dominant_frequency, spectral_entropy,
        spectral_centroid, spectral_rolloff, delta/theta/alpha/beta/gamma_power
    Complexity: hjorth_activity, hjorth_mobility, hjorth_complexity, sample_entropy
    Librosa (fs + librosa): mfcc → n_mfcc values
    """

    STATISTICAL = frozenset({
        'mean', 'std', 'var', 'skewness', 'kurtosis',
        'rms', 'peak_to_peak', 'zero_crossing_rate', 'line_length',
    })
    SPECTRAL = frozenset({
        'spectral_power', 'dominant_frequency', 'spectral_entropy',
        'spectral_centroid', 'spectral_rolloff',
        'delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power',
    })
    COMPLEXITY = frozenset({
        'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity', 'sample_entropy',
    })
    LIBROSA_FEATURES = frozenset({'mfcc'})

    def __init__(self, fs=1.0, n_mfcc=13, mfcc_hop_length=512,
                 sample_entropy_m=2, sample_entropy_r=None,
                 spectral_rolloff_pct=0.85, eeg_bands=None):
        """
        :param fs: sampling frequency in Hz (required for spectral features)
        :param n_mfcc: number of MFCC coefficients
        :param mfcc_hop_length: librosa hop length for MFCC
        :param sample_entropy_m: template length for sample entropy
        :param sample_entropy_r: tolerance (None → 0.2·std)
        :param spectral_rolloff_pct: cumulative power threshold for rolloff
        :param eeg_bands: dict overriding EEG_BANDS
        """
        self.fs = fs
        self.n_mfcc = n_mfcc
        self.mfcc_hop_length = mfcc_hop_length
        self.sample_entropy_m = sample_entropy_m
        self.sample_entropy_r = sample_entropy_r
        self.spectral_rolloff_pct = spectral_rolloff_pct
        self.eeg_bands = eeg_bands if eeg_bands is not None else EEG_BANDS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, signal, features=None):
        """Extract a 1-D feature vector from a single signal window.

        :param features: feature name list; None → all statistical + complexity
        :return: 1-D numpy array
        """
        x = np.asarray(signal, dtype=float)
        if features is None:
            features = sorted(self.STATISTICAL | self.COMPLEXITY)

        result = []
        for feat in features:
            val = self._dispatch(x, feat)
            if np.isscalar(val):
                result.append(float(val))
            else:
                result.extend(float(v) for v in val)
        return np.array(result, dtype=float)

    def extract_batch(self, signals, features=None):
        """Extract features from a 2-D batch of signal windows.

        :param signals: (n_windows, window_length) array-like
        :param features: feature name list; None → all statistical + complexity
        :return: (n_windows, n_features) numpy array
        """
        signals = np.asarray(signals, dtype=float)
        rows = [self.extract(signals[i], features) for i in range(signals.shape[0])]
        return np.stack(rows, axis=0)

    def feature_names(self, features=None):
        """Feature names in the same order as extract(); mfcc expands to mfcc_0…mfcc_N."""
        if features is None:
            features = sorted(self.STATISTICAL | self.COMPLEXITY)
        names = []
        for feat in features:
            if feat == 'mfcc':
                names += [f'mfcc_{i}' for i in range(self.n_mfcc)]
            else:
                names.append(feat)
        return names

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, x, feat):
        if feat == 'mean':
            return np.mean(x)
        if feat == 'std':
            return np.std(x)
        if feat == 'var':
            return np.var(x)
        if feat == 'skewness':
            return _skewness(x)
        if feat == 'kurtosis':
            return _kurtosis(x)
        if feat == 'rms':
            return np.sqrt(np.mean(x ** 2))
        if feat == 'peak_to_peak':
            return float(np.ptp(x))
        if feat == 'zero_crossing_rate':
            return zero_crossing_rate(x)
        if feat == 'line_length':
            return line_length(x)
        if feat == 'hjorth_activity':
            return hjorth_parameters(x)[0]
        if feat == 'hjorth_mobility':
            return hjorth_parameters(x)[1]
        if feat == 'hjorth_complexity':
            return hjorth_parameters(x)[2]
        if feat == 'sample_entropy':
            return sample_entropy(x, m=self.sample_entropy_m, r=self.sample_entropy_r)
        if feat == 'spectral_power':
            return spectral_power(x, self.fs)
        if feat == 'dominant_frequency':
            return dominant_frequency(x, self.fs)
        if feat == 'spectral_entropy':
            return spectral_entropy(x, self.fs)
        if feat == 'spectral_centroid':
            return spectral_centroid(x, self.fs)
        if feat == 'spectral_rolloff':
            return spectral_rolloff(x, self.fs, pct=self.spectral_rolloff_pct)
        if feat in ('delta_power', 'theta_power', 'alpha_power', 'beta_power', 'gamma_power'):
            band = feat.replace('_power', '')
            lo, hi = self.eeg_bands[band]
            return band_power(x, self.fs, lo, hi)
        if feat == 'mfcc':
            return compute_mfcc(x, self.fs, self.n_mfcc, self.mfcc_hop_length)
        raise ValueError(
            f"Unknown feature '{feat}'. Available: "
            f"{sorted(self.STATISTICAL | self.SPECTRAL | self.COMPLEXITY | self.LIBROSA_FEATURES)}"
        )


# ---------------------------------------------------------------------------
# Statistical features
# ---------------------------------------------------------------------------

def zero_crossing_rate(x):
    """Fraction of samples where the signal crosses zero."""
    return np.sum(np.abs(np.diff(np.sign(x)))) / (2.0 * (len(x) - 1))


def line_length(x):
    """Sum of absolute sample-to-sample differences; widely used in epilepsy detection."""
    return float(np.sum(np.abs(np.diff(x))))


def hjorth_parameters(x):
    """Hjorth activity, mobility, and complexity.

    Activity = variance; mobility ∝ mean frequency; complexity = similarity to a sine (=1).

    :return: (activity, mobility, complexity) tuple of floats
    """
    activity = float(np.var(x))
    dx = np.diff(x)
    var_dx = float(np.var(dx))
    mobility = float(np.sqrt(var_dx / activity)) if activity > 0 else 0.0
    d2x = np.diff(dx)
    var_d2x = float(np.var(d2x))
    mob_dx = float(np.sqrt(var_d2x / var_dx)) if var_dx > 0 else 0.0
    complexity = float(mob_dx / mobility) if mobility > 0 else 0.0
    return activity, mobility, complexity


def sample_entropy(x, m=2, r=None):
    """Regularity measure robust to signal length; lower = more predictable.

    O(N²·m); avoid on windows >2000 samples.

    :param m: template length (2 is standard)
    :param r: similarity tolerance (None → 0.2·std)
    """
    x = np.asarray(x, dtype=float)
    if r is None:
        r = 0.2 * np.std(x)
    N = len(x)

    def _count(length):
        count = 0
        for i in range(N - length):
            template = x[i:i + length]
            for j in range(i + 1, N - length):
                if np.max(np.abs(x[j:j + length] - template)) < r:
                    count += 1
        return count

    A = _count(m + 1)
    B = _count(m)
    if B == 0:
        return 0.0
    return float(-np.log(A / B))


def _skewness(x):
    if _SCIPY:
        return float(_sp_stats.skew(x))
    n = len(x)
    mu, sigma = np.mean(x), np.std(x)
    return 0.0 if sigma == 0 else float(np.sum((x - mu) ** 3) / (n * sigma ** 3))


def _kurtosis(x):
    if _SCIPY:
        return float(_sp_stats.kurtosis(x))
    n = len(x)
    mu, sigma = np.mean(x), np.std(x)
    return 0.0 if sigma == 0 else float(np.sum((x - mu) ** 4) / (n * sigma ** 4) - 3)


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

def _psd(x, fs):
    """PSD via Welch (scipy) or periodogram (numpy fallback)."""
    if _SCIPY:
        return _sp_signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(x)) ** 2) / n
    return freqs, psd


def spectral_power(x, fs):
    """Total signal power estimated from the PSD."""
    freqs, psd = _psd(x, fs)
    return float(_trapz(psd, freqs))


def dominant_frequency(x, fs):
    """Frequency at which the PSD is maximum."""
    freqs, psd = _psd(x, fs)
    return float(freqs[np.argmax(psd)])


def spectral_entropy(x, fs):
    """Shannon entropy of the normalised PSD; low for narrow-band, high for broadband."""
    _, psd = _psd(x, fs)
    total = psd.sum()
    if total == 0:
        return 0.0
    p = psd / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def spectral_centroid(x, fs):
    """Frequency-weighted mean of the PSD (centre of mass of the spectrum)."""
    freqs, psd = _psd(x, fs)
    total = psd.sum()
    return float(np.sum(freqs * psd) / total) if total > 0 else 0.0


def spectral_rolloff(x, fs, pct=0.85):
    """Frequency below which pct of total spectral power is contained."""
    freqs, psd = _psd(x, fs)
    cumsum = np.cumsum(psd)
    idx = np.searchsorted(cumsum, pct * cumsum[-1])
    return float(freqs[min(idx, len(freqs) - 1)])


def band_power(x, fs, f_low, f_high):
    """Integrate PSD over [f_low, f_high] Hz; standard for EEG rhythm analysis."""
    freqs, psd = _psd(x, fs)
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not mask.any():
        return 0.0
    return float(_trapz(psd[mask], freqs[mask]))


# ---------------------------------------------------------------------------
# Librosa-based features
# ---------------------------------------------------------------------------

def compute_mfcc(x, fs, n_mfcc=13, hop_length=512):
    """Mean MFCC coefficients over a window; returns length-n_mfcc vector. Requires librosa."""
    if not _LIBROSA:
        raise ImportError(
            "librosa is required for MFCC computation. "
            "Install with:  pip install librosa"
        )
    mfccs = _librosa.feature.mfcc(
        y=x.astype(float), sr=int(fs), n_mfcc=n_mfcc, hop_length=hop_length
    )
    return np.mean(mfccs, axis=1)
