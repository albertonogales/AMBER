"""Shared fixtures for the AMBER test suite."""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend before any pyplot import

import numpy as np
import plotly.io as pio
import pytest

import AMBER

# Force plotly to use a headless renderer — prevents iplot/notebook errors in CI
pio.renderers.default = "json"


# ---------------------------------------------------------------------------
# Raw data fixtures  (seeded for reproducibility)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_data():
    """30 samples × 4 features — fast training."""
    return np.random.default_rng(0).standard_normal((30, 4))


@pytest.fixture
def medium_data():
    """100 samples × 8 features."""
    return np.random.default_rng(1).standard_normal((100, 8))


@pytest.fixture
def signal_1d():
    """256-sample sine wave with mild noise."""
    t = np.linspace(0, 2 * np.pi, 256)
    return np.sin(t) + 0.1 * np.random.default_rng(2).standard_normal(256)


@pytest.fixture
def signal_batch():
    """20 windows × 128 samples each."""
    rng = np.random.default_rng(3)
    t = np.linspace(0, 2 * np.pi, 128)
    return np.stack([np.sin(t + rng.uniform(0, np.pi)) for _ in range(20)])


# ---------------------------------------------------------------------------
# Trained-object fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trained_map(small_data):
    return AMBER.Map(data=small_data, size=4, period=30)


@pytest.fixture
def classification(trained_map, small_data):
    return AMBER.Classification(trained_map, small_data)


@pytest.fixture
def temporal_analysis(classification):
    return AMBER.TemporalAnalysis(classification)


@pytest.fixture
def trained_temporal_map(small_data):
    return AMBER.TemporalMap(
        data=small_data, size=4, period=30,
        context_weight=0.5, context_influence=0.3,
    )
