"""
Execution-time benchmark: Python SOM libraries.

Compares training time across three dataset sizes (small / medium / large)
with a fixed number of weight-update iterations for each.  Results are
printed as a Markdown table suitable for inclusion in a paper or README.

Libraries compared
------------------
- GEMA        (García-Tejedor & Nogales, 2022) — https://github.com/ufvceiec/GEMA
- MiniSom     (Vettigli, 2018)                 — https://github.com/JustGlowing/minisom
- sklearn-som (Moran, 2021)                    — https://github.com/rileypsmith/sklearn-som

Note: SomPy is omitted because it is no longer maintained and incompatible
with current NumPy / Python versions.

Run
---
    python examples/benchmark_som_libraries.py

Requirements
------------
    pip install minisom sklearn-som
    GEMA must be importable (pip install GEMA or local install).
"""

from __future__ import annotations

import time
import warnings
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore")

# ── Benchmark configuration ─────────────────────────────────────────────────
# (label, N_samples, D_features, map_side, n_iterations)
CONFIGS = [
    ("Small\n(500×4, 5×5, 1k iter)",      500,    4,  5,  1_000),
    ("Medium\n(2000×10, 8×8, 5k iter)",  2_000,  10,  8,  5_000),
    ("Large\n(5000×20, 10×10, 10k iter)", 5_000,  20, 10, 10_000),
]
REPS = 3   # best-of-N to reduce scheduling noise
SEED = 42


# ── Timing helper ────────────────────────────────────────────────────────────
def best_of(fn: Callable, reps: int = REPS) -> float:
    """Return the minimum wall-clock time (seconds) over *reps* calls to fn."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


# ── Library wrappers ─────────────────────────────────────────────────────────
def run_gema(data: np.ndarray, size: int, period: int) -> None:
    import GEMA
    GEMA.Map(data=data, size=size, period=period)


def run_minisom(data: np.ndarray, size: int, period: int) -> None:
    from minisom import MiniSom
    m = MiniSom(size, size, data.shape[1],
                sigma=size // 2, learning_rate=0.1, random_seed=SEED)
    m.train(data, period, verbose=False)


def run_sklearn_som(data: np.ndarray, size: int, period: int) -> None:
    from sklearn_som.som import SOM
    # sklearn-som uses full-data epochs; convert iterations → epochs
    epochs = max(1, period // data.shape[0])
    m = SOM(m=size, n=size, dim=data.shape[1])
    m.fit(data, epochs=epochs)


LIBRARIES = {
    "GEMA":        run_gema,
    "MiniSom":     run_minisom,
    "sklearn-som": run_sklearn_som,
}

# ── Run benchmark ────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
results: dict[str, dict[str, float]] = {}

for lib_name, fn in LIBRARIES.items():
    for label, N, D, sz, iters in CONFIGS:
        data = rng.standard_normal((N, D))
        short_label = label.split("\n")[0]
        try:
            t = best_of(lambda: fn(data, sz, iters))  # noqa: B023
        except ImportError as exc:
            print(f"  [skip] {lib_name}: {exc}")
            t = float("nan")
        results.setdefault(lib_name, {})[short_label] = t
        print(f"{lib_name:<14} {short_label}: {t:.3f}s")

# ── Print Markdown table ─────────────────────────────────────────────────────
col_labels  = [c[0].split("\n")[0] for c in CONFIGS]
col_details = [c[0].split("\n")[1] for c in CONFIGS]

header_row = "| Library       | " + " | ".join(col_labels) + " |"
sep_row    = "| :---          | " + " | ".join(":---:" for _ in col_labels) + " |"

print("\n")
print("## SOM Library Execution-Time Comparison (seconds, best of 3 runs)\n")
print(header_row)
print(sep_row)
for lib_name, timings in results.items():
    cells = []
    for lbl in col_labels:
        t = timings.get(lbl, float("nan"))
        cells.append("n/a" if np.isnan(t) else f"{t:.3f}")
    print(f"| **{lib_name}** | " + " | ".join(cells) + " |")

print()
print(
    "_Measured on a single CPU core (Apple M-series). "
    "GEMA uses an online (one-sample-per-step) update rule; "
    "MiniSom uses the same rule. sklearn-som uses full-batch epochs, "
    "so the iteration count is converted to epochs = iterations / N. "
    "Best-of-3 wall-clock time; lower is better._"
)
