"""
Feature extraction for time series (biosignals, audio).

FeatureExtractor converts raw signal windows into fixed-length vectors for Map.train/classify.
Requires scipy (mandatory) and librosa (optional, only for 'mfcc').
"""

import numpy as np
from scipy import signal as _sp_signal
from scipy import stats as _sp_stats

# np.trapezoid introduced in NumPy 2.0; np.trapz removed in NumPy 2.0.
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')

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

    def extract(self, signal, features=None):
        """Extract a 1-D feature vector from a single signal window.

        :param features: feature name list; None → all statistical + complexity
        :return: 1-D numpy array
        """
        x = np.asarray(signal, dtype=float)
        if features is None:
            features = sorted(self.STATISTICAL | self.COMPLEXITY)

        cache = {}
        result = []
        for feat in features:
            val = self._dispatch(x, feat, cache)
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
        if signals.shape[0] == 0:
            n_feats = len(self.extract(np.zeros(signals.shape[1] if signals.ndim > 1 else 1), features))
            return np.empty((0, n_feats), dtype=float)
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

    def _build_dispatch(self):
        """Return a dict mapping feature name → callable(x, cache)."""
        _HJORTH = ('hjorth_activity', 'hjorth_mobility', 'hjorth_complexity')

        def _hjorth(idx):
            def fn(x, cache):
                if 'hjorth' not in cache:
                    cache['hjorth'] = hjorth_parameters(x)
                return cache['hjorth'][idx]
            return fn

        def _spectral(fn):
            def wrapped(x, cache):
                if 'psd' not in cache:
                    cache['psd'] = _psd(x, self.fs)
                freqs, psd = cache['psd']
                return fn(x, freqs, psd)
            return wrapped

        def _band(band_name):
            def fn(x, cache):
                if 'psd' not in cache:
                    cache['psd'] = _psd(x, self.fs)
                freqs, psd = cache['psd']
                lo, hi = self.eeg_bands[band_name]
                return band_power(x, self.fs, lo, hi, freqs=freqs, psd=psd)
            return fn

        table = {
            # Statistical
            'mean':               lambda x, c: np.mean(x),
            'std':                lambda x, c: np.std(x),
            'var':                lambda x, c: np.var(x),
            'skewness':           lambda x, c: _skewness(x),
            'kurtosis':           lambda x, c: _kurtosis(x),
            'rms':                lambda x, c: np.sqrt(np.mean(x ** 2)),
            'peak_to_peak':       lambda x, c: float(np.max(x) - np.min(x)),
            'zero_crossing_rate': lambda x, c: zero_crossing_rate(x),
            'line_length':        lambda x, c: line_length(x),
            # Hjorth (shared computation)
            'hjorth_activity':    _hjorth(0),
            'hjorth_mobility':    _hjorth(1),
            'hjorth_complexity':  _hjorth(2),
            # Complexity
            'sample_entropy':     lambda x, c: sample_entropy(
                                      x, m=self.sample_entropy_m, r=self.sample_entropy_r),
            # Spectral (shared PSD computation)
            'spectral_power':     _spectral(lambda x, f, p: spectral_power(x, self.fs, freqs=f, psd=p)),
            'dominant_frequency': _spectral(lambda x, f, p: dominant_frequency(x, self.fs, freqs=f, psd=p)),
            'spectral_entropy':   _spectral(lambda x, f, p: spectral_entropy(x, self.fs, psd=p)),
            'spectral_centroid':  _spectral(lambda x, f, p: spectral_centroid(x, self.fs, freqs=f, psd=p)),
            'spectral_rolloff':   _spectral(lambda x, f, p: spectral_rolloff(
                                      x, self.fs, pct=self.spectral_rolloff_pct, freqs=f, psd=p)),
            # EEG band powers (shared PSD computation)
            **{f'{b}_power': _band(b) for b in self.eeg_bands},
            # Librosa
            'mfcc':               lambda x, c: compute_mfcc(
                                      x, self.fs, self.n_mfcc, self.mfcc_hop_length),
        }
        return table

    def _dispatch(self, x, feat, cache=None):
        if cache is None:
            cache = {}
        if '_table' not in self.__dict__:
            self.__dict__['_table'] = self._build_dispatch()
        fn = self.__dict__['_table'].get(feat)
        if fn is None:
            raise ValueError(
                f"Unknown feature '{feat}'. Available: "
                f"{sorted(self.STATISTICAL | self.SPECTRAL | self.COMPLEXITY | self.LIBROSA_FEATURES)}"
            )
        return fn(x, cache)


def zero_crossing_rate(x):
    """Fraction of samples where the signal crosses zero."""
    return np.sum(np.abs(np.diff(np.sign(x)))) / (2.0 * (len(x) - 1))


def line_length(x):
    """Sum of absolute sample-to-sample differences; widely used in epilepsy detection."""
    return float(np.sum(np.abs(np.diff(x))))


def hjorth_parameters(x):
    """Hjorth (1970) activity, mobility, and complexity. Returns a 3-tuple of floats."""
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
    """Sample entropy (Richman & Moorman 2000). Lower = more regular. O(N²·m).

    :param m: template length (2 is standard)
    :param r: similarity tolerance (None → 0.2·std); B==0 → returns np.inf
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
        return np.inf   # no template matches at length m → maximally irregular
    return float(-np.log(A / B))


def _skewness(x):
    return float(_sp_stats.skew(x))


def _kurtosis(x):
    return float(_sp_stats.kurtosis(x))


def _psd(x, fs):
    return _sp_signal.welch(x, fs=fs, nperseg=min(256, len(x)))


def spectral_power(x, fs, freqs=None, psd=None):
    """Total signal power estimated from the PSD."""
    if freqs is None or psd is None:
        freqs, psd = _psd(x, fs)
    return float(_trapz(psd, freqs))


def dominant_frequency(x, fs, freqs=None, psd=None):
    """Frequency at which the PSD is maximum."""
    if freqs is None or psd is None:
        freqs, psd = _psd(x, fs)
    return float(freqs[np.argmax(psd)])


def spectral_entropy(x, fs, psd=None):
    """Shannon entropy of the normalised PSD; low for narrow-band, high for broadband."""
    if psd is None:
        _, psd = _psd(x, fs)
    total = psd.sum()
    if total == 0:
        return 0.0
    p = psd / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def spectral_centroid(x, fs, freqs=None, psd=None):
    """Frequency-weighted mean of the PSD (centre of mass of the spectrum)."""
    if freqs is None or psd is None:
        freqs, psd = _psd(x, fs)
    total = psd.sum()
    return float(np.sum(freqs * psd) / total) if total > 0 else 0.0


def spectral_rolloff(x, fs, pct=0.85, freqs=None, psd=None):
    """Frequency below which pct of total spectral power is contained."""
    if freqs is None or psd is None:
        freqs, psd = _psd(x, fs)
    cumsum = np.cumsum(psd)
    idx = np.searchsorted(cumsum, pct * cumsum[-1])
    return float(freqs[min(idx, len(freqs) - 1)])


def band_power(x, fs, f_low, f_high, freqs=None, psd=None):
    """Integrate PSD over [f_low, f_high] Hz; standard for EEG rhythm analysis."""
    if freqs is None or psd is None:
        freqs, psd = _psd(x, fs)
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not mask.any():
        return 0.0
    return float(_trapz(psd[mask], freqs[mask]))


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
