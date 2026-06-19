# AMBER

**AMBER** is an open-source Python library for building, training, and analysing **Self-Organizing Maps (SOMs)**.  
It covers the full workflow — data normalisation, map training, classification, temporal/recurrent SOMs, feature extraction for biosignals, and interactive visualisation.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Tests](https://github.com/albertonogales/AMBER/actions/workflows/tests.yml/badge.svg)](https://github.com/albertonogales/AMBER/actions/workflows/tests.yml)
[![Coverage](https://codecov.io/gh/albertonogales/AMBER/branch/main/graph/badge.svg)](https://codecov.io/gh/albertonogales/AMBER)
[![Docs](https://readthedocs.org/projects/amber-som/badge/?version=latest)](https://amber-som.readthedocs.io/en/latest/)

---

## Features

| Module | What it provides |
|--------|-----------------|
| `Map` | Standard SOM with 7 distance metrics, 9 normalisation strategies, 4 weight-initialisation methods, and automatic map-size selection (Vesanto & Alhoniemi 2000) |
| `Classification` | BMU assignment, activation map, U-matrix, quantisation error, topological error |
| `TemporalMap` | Recurrent SOM (RSOM, Voegtlin 2002) — context vector accumulates temporal structure |
| `TemporalAnalysis` | Transition matrix, stability index, mean path length, dwell times |
| `FeatureExtractor` | Statistical, spectral, and complexity features for time-series/biosignals |
| `IterativeSOM` | Grid-search over map sizes; selects the best map by topological error |
| `Visualization` | Heat maps, U-matrix, elevation maps, trajectory plots, transition heatmaps |

### Distance metrics (BMU selection)

`euclidean` · `manhattan` · `chebyshev` · `cosine` · `correlation` · `dtw` · `cross_correlation`

### Normalisation strategies

`zscore` (fwn) · `robust` · `01scale` · `zscore_sample` · `robust_sample` · `minmax_sample` · `l2` (euclidean) · `none`

---

## Installation

### From PyPI (recommended)

```bash
pip install amber-som
```

### From source

```bash
git clone https://github.com/albertonogales/AMBER.git
cd AMBER
pip install -e .
```

### Optional dependencies

| Extra | Install command | Required for |
|-------|----------------|-------------|
| scipy | `pip install scipy` | Welch PSD, improved skewness/kurtosis |
| librosa | `pip install librosa` | MFCC feature extraction |

---

## Quick start

```python
import numpy as np
import AMBER

# --- 1. Train a SOM (map size chosen automatically) ---
data = np.random.rand(200, 8)
som = AMBER.Map(data=data, period=100)
print(f"Map size: {som.map_size}×{som.map_size}")   # e.g. 10×10

# --- 2. Classify ---
cls = AMBER.Classification(som, data)
print(f"Quantisation error : {cls.quantization_error:.4f}")
print(f"Topological error  : {cls.topological_error:.4f}")

# --- 3. Visualise ---
AMBER.Visualization.heat_map(cls)
AMBER.Visualization.umatrix(cls)
```

### Custom map size and distance

```python
som = AMBER.Map(
    data=data,
    size=12,
    period=200,
    distance='dtw',          # DTW distance for BMU selection
    normalization='zscore',
    weights='PCA',
    use_decay=True,
)
```

### Temporal / Recurrent SOM

```python
# Train
tsom = AMBER.TemporalMap(
    data=eeg_windows,
    period=100,
    context_weight=0.5,     # α — context memory
    context_influence=0.3,  # β — context vs. signal balance
)

# Analyse temporal dynamics
cls  = AMBER.Classification(tsom, eeg_windows)
ta   = AMBER.TemporalAnalysis(cls)
print(ta.summary())

AMBER.Visualization.trajectory(cls, ta)
AMBER.Visualization.transition_matrix_plot(ta)
AMBER.Visualization.dwell_time_map(ta, cls)
```

### Feature extraction for biosignals

```python
from AMBER import FeatureExtractor

fe = FeatureExtractor(fs=256)            # 256 Hz EEG

# Single window → 1-D feature vector
x = fe.extract(eeg_window, features=[
    'rms', 'zero_crossing_rate',
    'alpha_power', 'spectral_entropy',
    'hjorth_mobility',
])

# Batch of windows → (n_windows, n_features) ready for Map.train
X = fe.extract_batch(eeg_windows, features=['rms', 'alpha_power', 'beta_power'])
```

---

## Application domains

AMBER has been applied to and includes example notebooks for:

| Domain | Signal type | Key AMBER feature |
|--------|-------------|-------------------|
| **EEG / biosignals** | Brain, muscle, cardiac | `FeatureExtractor` (Hjorth, band power) + `TemporalMap` for state transitions — see `eeg_som.ipynb` |
| **Audio / speech** | MFCC, spectral features | `FeatureExtractor(mfcc)` + `Map` for speaker/instrument clustering — see `audio_som.ipynb` |
| **Climate** | SST, MSLP, wind | `TemporalMap` for weather-regime discovery and transition analysis |
| **Predictive maintenance** | Vibration (bearing) | `FeatureExtractor` (kurtosis, RMS) + degradation trajectory via `TemporalAnalysis` |
| **Finance / HAR** | Returns, accelerometers | `TemporalMap` for regime / activity detection |

---

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest               # 372 tests, ~99 % coverage
```

A coverage HTML report is written to `coverage_html/`.

---

## Repository structure

```
AMBER/
├── AMBER/
│   ├── __init__.py
│   ├── map.py                 # Core SOM
│   ├── classification.py      # BMU assignment & metrics
│   ├── distances.py           # All signal and grid distance functions
│   ├── features.py            # Feature extraction for time series
│   ├── temporal_map.py        # Recurrent SOM (RSOM)
│   ├── temporal_analysis.py   # Temporal metrics
│   ├── iterativesom.py        # Grid-search over map sizes
│   └── visualization.py       # Plots
├── tests/
│   ├── conftest.py
│   ├── test_distances.py
│   ├── test_map.py
│   ├── test_classification.py
│   ├── test_features.py
│   ├── test_temporal_map.py
│   ├── test_temporal_analysis.py
│   ├── test_iterativesom.py
│   ├── test_visualization.py
│   └── test_integration.py
├── examples/
│   ├── colors_example.ipynb              # Basic SOM on RGB colour data
│   ├── eeg_som.ipynb                     # EEG sleep staging (Sleep-EDF)
│   ├── audio_som.ipynb                   # Audio/instrument clustering (NSynth/FSDD)
│   ├── climate_som.ipynb                 # Weather-regime discovery (ERA5)
│   ├── predictive_maintenance_som.ipynb  # Bearing degradation tracking (CWRU)
│   └── ablation_study.ipynb              # Systematic validation of all design choices
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── setup.cfg
└── LICENSE
```

---

## Citing AMBER

If you use AMBER in your research, please cite the companion paper:

```bibtex
@article{nogales2026amber,
  title   = {{AMBER}: Autoassociative Map Builder for tEmporal Representations —
             a Python library for Self-Organizing Maps},
  author  = {Nogales, Alberto and Sicilia, Miguel {\'A}ngel and
             Garc{\'i}a Barriocanal, Elena and Mora Cantallops, Mar{\c{c}}al and
             Ballesteros Rodr{\'i}guez, Alberto},
  journal = {Information Sciences},
  year    = {2026},
  note    = {Under review}
}
```

A machine-readable `CITATION.cff` file is included in the repository for
automated citation tools (GitHub "Cite this repository", Zenodo, etc.).


---

## Contact

- **Responsible:** Alberto Nogales — alberto.nogales@uah.es  
- **Supervisors:** Alberto Nogales, Miguel Ángel Sicilia  


Under license of IERU Research Group at Universidad de Alcalá (Spain)
