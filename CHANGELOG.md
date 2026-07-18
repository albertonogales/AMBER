# Changelog

All notable changes to AMBER are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.2.0] — 2026

### Fixed
- **`__trainned` typo** renamed to `__trained` throughout `map.py` — was visible in serialised JSON and test introspection.
- **`reinforce()` neighbourhood bug**: neighbourhood radius now continues from `1.0` in rounds 2+, instead of resetting to the initial value each round; introduces `round_neighbourhood` local variable.
- **Linear LR decay to zero**: denominator changed from `T` to `T+1` so the final training iteration receives a small but non-zero learning rate (previously a wasted no-op update).
- **Robbins–Monro claim**: added citation `(Robbins & Monro, 1951, Ann. Math. Stat. 22(3):400-407)` to the `asymptotic` LR docstring.
- **`trajectory()` randomness**: added `:param random_seed:` docstring entry explaining the jitter RNG.
- **`IterativeSOM` model-selection bias**: emits `logger.warning` when `give_best=True` and `validation_data=None` (in-sample selection).

### Added
- `CITATION.cff` — machine-readable citation metadata for GitHub, Zenodo, and automated citation tools.
- Comprehensive citation block in `README.md` (BibTeX entry, CFF pointer).
- 27 additional tests: scipy-absent fallbacks, librosa MFCC paths, `reinforce()` sequential branch, `TemporalMap.reinforce()`, `TemporalMap` norm_params save/load, `Classification` `verbose=True` and `other` DataFrame, visualization save-to-file paths, `trajectory` umatrix background, `IterativeSOM` selection-bias warning.
- 2 additional tests for `dtw_distance` `RuntimeWarning` when sequence length exceeds threshold (374 tests, 99% coverage).
- scipy promoted from optional to required dependency; `spectral` extra removed.
- `RuntimeWarning` added to `dtw_distance` and `Map.train` when DTW is used without a Sakoe-Chiba band on long sequences.
- `period` parameter semantics clarified with runtime log showing equivalent epoch count.

### Changed
- `CONTRIBUTING.md`: corrected GitHub URL (`ufvceiec` → `albertonogales`) and contact email (`@ceiec.es` → `@uah.es`); coverage threshold updated to 99%.
- `docs/getting_started.rst`: Python requirement updated to 3.9+; test count updated to 374, 99%.
- `README.md`: test count updated to 374, 99%.
- `run_cbf_som.py` moved from project root to `examples/cbf_dtw_example.py`.

---

## [2.1.0] — 2026

### Added
- `random_seed` parameter on `Map` and `TemporalMap` — deterministic training; persisted in JSON save/load.
- Full PEP 484 type annotations on all public methods (`from __future__ import annotations`).
- `ruff` and `mypy` added to CI and `requirements-dev.txt` — linting and type-checking on every push.
- `[tool.ruff]` and `[tool.mypy]` config sections in `pyproject.toml`.
- 12 new tests: `TestReproducibility`, `TestReinforce`, `TestSaveLoadExtended`, `TestTemporalMapReproducibility` (315 tests, 96% coverage).

### Changed
- All `print()` calls replaced with `logging.getLogger(__name__)` — library no longer writes to stdout.
- Packaging consolidated to a single `pyproject.toml` (PEP 517/518); `setup.py` deleted; `requires-python = ">=3.9"`.
- `Classification.verbose` changed from `int` to `bool = False`.
- Distortion and Euclidean QE computations vectorised (NumPy broadcasting, no Python loop over N).
- CI matrix updated from Python 3.8–3.11 to **3.9–3.12** (dropped EOL 3.8).
- Input validation uses `raise ValueError` instead of `assert` (guards survive `python -O`).
- `period` docstring clarified: total pattern presentations, not full epochs (`period ≈ K × N` for K epochs).
- `reinforce()` neighbourhood now decays via `variation_neighbourhood` (was hardcoded to 1).
- `reinforce()` now passes `mode=self.lr_decay` to `variation_learning_rate`.
- Removed duplicate `self.presentation` assignment in `Map.__init__`.
- `cross_correlation_distance` docstring updated with formal Cauchy-Schwarz proof that return value ∈ [0, 1].
- README badges updated to Python 3.9+, 96% coverage.

---

## [2.0.0] — 2024

Major revision of the original GEMA library, renamed to **AMBER**.

### Added
- **Seven signal distance metrics** for BMU selection: `euclidean`, `manhattan`, `chebyshev`, `cosine`, `correlation`, `dtw` (with optional Sakoe-Chiba band), `cross_correlation` — all wired through a registry so the `distance` parameter is fully honoured during training.
- **Nine normalisation strategies**: `zscore`/`fwn` (per-feature), `robust`, `01scale`, `zscore_sample`, `robust_sample`, `minmax_sample`, `l2`/`euclidean`, `none`.
- **Automatic map-size selection** via the Vesanto & Alhoniemi (2000) heuristic (`size = max(2, round(√(5·√N)))`) when `size` is not provided by the user.
- **`FeatureExtractor`** class for time-series / biosignal feature extraction: statistical (RMS, ZCR, line length, Hjorth parameters, skewness, kurtosis, sample entropy), spectral (band power, spectral entropy, spectral centroid, spectral rolloff, EEG band powers), and MFCC (requires librosa).
- **`TemporalMap`** — Recurrent SOM (Voegtlin 2002 RSOM): context vector `c_t = α·c_{t-1} + (1-α)·w_BMU` combined with signal distance weighted by `context_influence` (β).
- **`TemporalAnalysis`** — transition matrix, row-normalised transition probabilities, stability index, mean path length, dwell times per neuron.
- **Temporal visualisations**: `Visualization.trajectory`, `Visualization.transition_matrix_plot`, `Visualization.dwell_time_map`.
- **Complete test suite**: 265 tests across 9 modules, ≥95% coverage, using pytest + pytest-cov.
- `vesanto_size(n)` exposed at the package level as `AMBER.vesanto_size`.
- `AMBER.AVAILABLE_DISTANCES` list exposed at the package level.
- `distances.py` module with all scalar and matrix distance functions plus grid distance functions.

### Fixed
- `assert initial_lr < 1 or initial_lr > 0` was always `True`; corrected to `assert 0 < initial_lr < 1`.
- `is not 0` / `is not 'none'` identity comparisons replaced with `!= 0` / `!= 'none'`.
- Euclidean normalisation loop had an off-by-one error; replaced with vectorised `np.linalg.norm`.
- `np.linalg.eig` on symmetric covariance matrix replaced with `np.linalg.eigh` (PCA init).
- `reinforcement_lr` was computed but never used in the training loop.
- Topological error excluded diagonal neighbours; corrected to Chebyshev grid distance > 1.
- `IterativeSOM` was non-functional; fully rewritten.
- Chained-assignment `FutureWarning` in `classification.py` fixed with `.loc[]` indexing.
- `np.trapz` (deprecated) replaced with `np.trapezoid`.
- `cm.get_cmap` (deprecated) replaced with `plt.colormaps.get_cmap`.
- `distance` parameter was ignored during BMU calculation (always used squared Euclidean); now fully functional.

### Changed
- Library renamed from **GEMA** to **AMBER**.
- `Map.__init__` parameter `size` defaults to `None` (was `-1`); auto-size triggers when omitted.
- `save_classifier` / `load_classifier` now persist `normalization` and `weights_init` fields.
- Grid-space distance (neighbourhood update) decoupled from signal-space distance (BMU search) via `_grid_distance`.

---

## [1.0.0] — 2022

Original GEMA release accompanying the Software Impacts paper:

> García-Tejedor, Á. J., & Nogales, A. (2022). An Open-Source Python Library for Self-Organizing-Maps. *Software Impacts*, 12, 100280. https://doi.org/10.1016/j.simpa.2022.100280
