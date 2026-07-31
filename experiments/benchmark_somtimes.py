"""
AMBER vs SOMtimes — Benchmark Experiment
=========================================
Compares the AMBER SOM library (v2.1.0) against SOMtimes (v1.0.2,
Ali Javed 2020) on two tasks:

  TASK 1 – Head-to-head comparison (same grid, same epochs)
    Dataset  : 80 synthetic time-series samples × 15 time points
               (4 classes: sine / square / sawtooth / triangle, σ=0.10)
    Grid     : 5×5 = 25 neurons  (Vesanto-5√80 ≈ 45, capped to 5×5 here
               so SOMtimes pure-Python DTW finishes in under 5 minutes)
    Epochs   : 5
    Distance : Euclidean (AMBER) vs windowSize=0 (SOMtimes intended)
               Note: SOMtimes windowSize=0 still falls through to
               constrained DTW (window=1) because LB-Keogh is always
               0 ≤ Euclidean distance, so the pruning condition is never
               satisfied.  This is flagged as a design issue.

  TASK 2 – Scalability stress test (AMBER only; SOMtimes estimated)
    Dataset  : 400 × 30 and 900 × 60 (see extrapolation note)
    Grid     : 20×20 (Vesanto heuristic)
    Epochs   : 30
    Distance : Euclidean and DTW (band=3)

Metrics
-------
  QE  – mean Euclidean BMU distance per sample (lower is better)
  TE  – topological error, 8-connected (Villmann 1997), lower is better
  t   – wall-clock training time in seconds (lower is better)
  std – std of QE over 5 independent runs (lower = more reproducible)

Runtime environment
-------------------
  dtaidistance C extension : NOT AVAILABLE on this machine
  SOMtimes patch applied   : dtw.distance_fast → dtw.distance (Python)
                             np.asarray inhomogeneous fix applied
  These patches are noted explicitly in the report so readers know the
  comparison uses SOMtimes' Python fallback, not the optimised C path.
"""

from __future__ import annotations

import sys
import os
import time
import warnings
import math

import numpy as np

# ── locate AMBER ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_AMBER_ROOT = os.path.join(_HERE, "..")
if _AMBER_ROOT not in sys.path:
    sys.path.insert(0, _AMBER_ROOT)

import AMBER

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def vesanto_grid(n: int, cap: int = 20) -> int:
    return min(cap, max(3, round(5 * math.sqrt(n))))


def compute_qe_te(weights_flat: np.ndarray, grid_shape: tuple,
                  data: np.ndarray) -> tuple[float, float]:
    """QE (Euclidean) and TE (8-connected) from flat weight array."""
    H, W = grid_shape
    w = weights_flat.reshape(H, W, -1)
    n, qe_acc, te_count = data.shape[0], 0.0, 0
    for x in data:
        dist_map = np.sqrt(np.sum((w - x) ** 2, axis=-1))
        flat = dist_map.flatten()
        i1 = int(np.argmin(flat)); flat[i1] = np.inf
        i2 = int(np.argmin(flat))
        qe_acc += dist_map.flatten()[i1]
        r1, c1 = divmod(i1, W); r2, c2 = divmod(i2, W)
        if max(abs(r1 - r2), abs(c1 - c2)) > 1:
            te_count += 1
    return qe_acc / n, te_count / n


# ─────────────────────────────────────────────────────────────────────────────
# Dataset generators
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic(n_per_class: int, length: int,
                   noise: float = 0.10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """4-class synthetic: sine / square / sawtooth / triangle."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, length)
    shapes = [
        np.sin(t),
        np.sign(np.sin(t)).astype(float),
        (t / (2 * np.pi)) * 2 - 1,
        2 * np.abs(t / np.pi - np.floor(t / np.pi + 0.5)),
    ]
    X, y = [], []
    for cls, proto in enumerate(shapes):
        X.append(proto + rng.normal(0, noise, (n_per_class, length)))
        y.extend([cls] * n_per_class)
    return np.vstack(X).astype(float), np.array(y, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# AMBER runner
# ─────────────────────────────────────────────────────────────────────────────

def run_amber(data: np.ndarray, grid: int, epochs: int,
              distance: str, dtw_band: int, seed: int) -> dict:
    t0 = time.perf_counter()
    som = AMBER.Map(
        data=data, size=grid, period=epochs,
        distance=distance,
        dtw_band=dtw_band if distance == 'dtw' else None,
        normalization='none',
        random_seed=seed,
    )
    elapsed = time.perf_counter() - t0
    clf = AMBER.Classification(som=som, classification_data=data)
    return {'time_s': elapsed, 'qe': clf.quantization_error_euclidean,
            'te': clf.topological_error}


# ─────────────────────────────────────────────────────────────────────────────
# SOMtimes runner
# ─────────────────────────────────────────────────────────────────────────────

def run_somtimes(data: np.ndarray, grid: int, epochs: int,
                 window_size: int) -> dict:
    from somtimes.SelfOrganizingMap import SelfOrganizingMap
    H = W = grid
    som = SelfOrganizingMap(inputSize=data.shape[1], hiddenSize=[H, W])
    t0 = time.perf_counter()
    som.iterate(inputs=data.tolist(), epochs=epochs, windowSize=window_size,
                k=1, randomInitilization=True)
    elapsed = time.perf_counter() - t0
    qe, te = compute_qe_te(som.getWeights(), (H, W), data)
    return {'time_s': elapsed, 'qe': qe, 'te': te}


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility (5 runs, Euclidean only)
# ─────────────────────────────────────────────────────────────────────────────

def repr_amber(data, grid, epochs, n=5):
    qes = [run_amber(data, grid, epochs, 'euclidean', 0, 42)['qe'] for _ in range(n)]
    return float(np.mean(qes)), float(np.std(qes))

def repr_somtimes(data, grid, epochs, n=5):
    qes = [run_somtimes(data, grid, epochs, 0)['qe'] for _ in range(n)]
    return float(np.mean(qes)), float(np.std(qes))


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — Head-to-head (small dataset)
# ─────────────────────────────────────────────────────────────────────────────

SEED         = 42
HH_N_CLASS   = 20     # 20 × 4 = 80 samples
HH_LENGTH    = 15
HH_GRID      = 5
HH_EPOCHS    = 5
HH_REPR_RUNS = 5


def task1():
    print("\n" + "="*60)
    print("TASK 1 — Head-to-head  (80×15,  5×5 grid,  5 epochs)")
    print("="*60)

    data, _ = make_synthetic(HH_N_CLASS, HH_LENGTH, noise=0.10, seed=SEED)
    results = {}

    # AMBER – Euclidean
    print("\n[AMBER — Euclidean]")
    r = run_amber(data, HH_GRID, HH_EPOCHS, 'euclidean', 0, SEED)
    results['amber_eucl'] = r
    print(f"  t={r['time_s']:.4f}s   QE={r['qe']:.5f}   TE={r['te']:.4f}")

    # SOMtimes – windowSize=0  (falls through to DTW-window-1 internally)
    print("[SOMtimes — windowSize=0 (Euclidean-intended; uses DTW-w1 internally)]")
    r = run_somtimes(data, HH_GRID, HH_EPOCHS, window_size=0)
    results['somtimes_eucl'] = r
    print(f"  t={r['time_s']:.4f}s   QE={r['qe']:.5f}   TE={r['te']:.4f}")

    # AMBER – DTW
    print("\n[AMBER — DTW (band=3)]")
    r = run_amber(data, HH_GRID, HH_EPOCHS, 'dtw', 3, SEED)
    results['amber_dtw'] = r
    print(f"  t={r['time_s']:.4f}s   QE={r['qe']:.5f}   TE={r['te']:.4f}")

    # SOMtimes – windowSize=3
    print("[SOMtimes — windowSize=3 (DTW Sakoe-Chiba band)]")
    r = run_somtimes(data, HH_GRID, HH_EPOCHS, window_size=3)
    results['somtimes_dtw'] = r
    print(f"  t={r['time_s']:.4f}s   QE={r['qe']:.5f}   TE={r['te']:.4f}")

    # Reproducibility
    print("\n[Reproducibility — 5 runs, Euclidean]")
    print("  AMBER …", end=' ', flush=True)
    mu_a, std_a = repr_amber(data, HH_GRID, HH_EPOCHS, n=HH_REPR_RUNS)
    results['amber_repr'] = {'mean': mu_a, 'std': std_a}
    print(f"mean={mu_a:.5f}  std={std_a:.2e}")

    print("  SOMtimes …", end=' ', flush=True)
    mu_s, std_s = repr_somtimes(data, HH_GRID, HH_EPOCHS, n=HH_REPR_RUNS)
    results['somtimes_repr'] = {'mean': mu_s, 'std': std_s}
    print(f"mean={mu_s:.5f}  std={std_s:.2e}")

    return results, data


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Scalability (AMBER only, SOMtimes extrapolated)
# ─────────────────────────────────────────────────────────────────────────────

SCALE_EPOCHS = 30

def task2(t1_data: np.ndarray, t1_somtimes_eucl: dict):
    """Run AMBER on large datasets; extrapolate SOMtimes from Task 1 timings."""

    print("\n" + "="*60)
    print("TASK 2 — Scalability stress test")
    print("="*60)

    configs = [
        ("Synth-4cl  400×30",  make_synthetic(100, 30, seed=SEED), vesanto_grid(400)),
        ("Synth-4cl  900×60",  make_synthetic(225, 60, seed=SEED), vesanto_grid(900)),
    ]

    # SOMtimes per-DTW-call time estimate from Task 1
    # Task 1: HH_N_CLASS*4 samples × HH_GRID^2 neurons × HH_EPOCHS epochs → t1_time
    t1_dtw_calls = (HH_N_CLASS * 4) * (HH_GRID ** 2) * HH_EPOCHS
    t1_time      = t1_somtimes_eucl['time_s']
    t_per_call   = t1_time / t1_dtw_calls  # seconds per DTW call (Python fallback)

    results = []
    for label, (data, _), grid in configs:
        n, T = data.shape
        print(f"\n  {label}  (grid={grid}×{grid})")

        # AMBER Euclidean
        r_eucl = run_amber(data, grid, SCALE_EPOCHS, 'euclidean', 0, SEED)
        print(f"  AMBER Eucl   t={r_eucl['time_s']:.3f}s  QE={r_eucl['qe']:.5f}  TE={r_eucl['te']:.4f}")

        # AMBER DTW
        r_dtw = run_amber(data, grid, SCALE_EPOCHS, 'dtw', 3, SEED)
        print(f"  AMBER DTW    t={r_dtw['time_s']:.3f}s  QE={r_dtw['qe']:.5f}  TE={r_dtw['te']:.4f}")

        # SOMtimes extrapolated
        est_calls = n * (grid ** 2) * SCALE_EPOCHS
        est_t     = est_calls * t_per_call
        print(f"  SOMtimes     estimated t≈{est_t:.0f}s ({est_t/3600:.1f}h)  "
              f"(extrapolated from Task-1 DTW rate: {t_per_call*1000:.3f} ms/call)")

        # Speed-up ratio
        speedup = est_t / r_eucl['time_s']
        print(f"  Speed-up AMBER vs SOMtimes ≈ ×{speedup:.0f}")

        results.append({
            'label': label, 'N': n, 'T': T, 'grid': grid,
            'amber_eucl': r_eucl, 'amber_dtw': r_dtw,
            'somtimes_est_t': est_t, 'speedup': speedup,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def sep(n=70): print("-" * n)

def print_task1_table(r):
    print("\n\n" + "="*70)
    print("TABLE 1 — Head-to-head (80×15, 5×5 grid, 5 epochs)")
    print("="*70)
    print(f"{'Metric':<20} {'AMBER':>12} {'SOMtimes':>12} {'Winner':>12}")
    sep()

    rows = [
        ("Euclidean  QE",   r['amber_eucl']['qe'],    r['somtimes_eucl']['qe']),
        ("Euclidean  TE",   r['amber_eucl']['te'],    r['somtimes_eucl']['te']),
        ("Euclidean  t(s)", r['amber_eucl']['time_s'],r['somtimes_eucl']['time_s']),
        ("DTW        QE",   r['amber_dtw']['qe'],     r['somtimes_dtw']['qe']),
        ("DTW        TE",   r['amber_dtw']['te'],     r['somtimes_dtw']['te']),
        ("DTW        t(s)", r['amber_dtw']['time_s'], r['somtimes_dtw']['time_s']),
        ("Repr. QE std",    r['amber_repr']['std'],   r['somtimes_repr']['std']),
    ]
    for label, av, sv in rows:
        w = "✓ AMBER" if av < sv else ("✓ SOMtimes" if sv < av else "tie")
        print(f"  {label:<18} {av:>12.5f} {sv:>12.5f} {w:>12}")
    sep()


def print_task2_table(results):
    print("\n\nTABLE 2 — Scalability (AMBER measured; SOMtimes extrapolated)")
    print("="*70)
    print(f"{'Dataset':<22} {'AMBER Eucl t':>14} {'AMBER DTW t':>12} {'SMT est. t':>14} {'Speed-up':>10}")
    sep()
    for r in results:
        print(f"  {r['label']:<20} {r['amber_eucl']['time_s']:>14.3f} "
              f"{r['amber_dtw']['time_s']:>12.3f} "
              f"{r['somtimes_est_t']:>12.0f}s "
              f"{r['speedup']:>9.0f}×")
    sep()


# ─────────────────────────────────────────────────────────────────────────────
# Feature comparison table (qualitative)
# ─────────────────────────────────────────────────────────────────────────────

def print_feature_table():
    print("\n\nTABLE 3 — Feature comparison (qualitative)")
    print("="*70)
    features = [
        ("Random seed control",            "Yes (random_seed=)",    "No"),
        ("Distance metrics",               "Eucl / DTW / Cosine / XCorr", "DTW only (windowSize=0 → DTW-w1)"),
        ("Normalization",                  "none/01/zscore/minmax", "None"),
        ("Quantization Error metric",      "Yes",                   "No"),
        ("Topological Error metric",       "Yes (4- or 8-conn.)",   "No"),
        ("U-Matrix",                       "Yes",                   "Yes (basic)"),
        ("Distortion measure",             "Yes (Graepel 1997)",    "No"),
        ("Model save / load (JSON)",       "Yes",                   "Numpy .npy"),
        ("Classification class",          "Yes (out-of-sample)",   "No"),
        ("Temporal / Recurrent SOM",       "Yes (TemporalMap)",     "No"),
        ("IterativeSOM (model selection)", "Yes",                   "No"),
        ("Feature extraction",             "Yes (FeatureExtractor)","No"),
        ("Pure-Python (no C ext.)",        "Yes",                   "No (requires dtaidistance C)"),
        ("Compiled C extension needed",    "No",                    "Yes (dtaidistance)"),
        ("Vectorised weight update",       "Yes (NumPy)",           "No (Python loop)"),
        ("Progress bar",                   "Yes (tqdm)",            "Epoch print only"),
        ("Test suite",                     "Yes (pytest)",          "No"),
        ("pip install",                    "amber-som",             "somtimes"),
    ]
    print(f"  {'Feature':<38} {'AMBER':>12}  {'SOMtimes'}")
    sep()
    for feat, av, sv in features:
        print(f"  {feat:<38} {av:>12}  {sv}")
    sep()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t1_results, t1_data = task1()
    t2_results = task2(t1_data, t1_results['somtimes_eucl'])

    print_task1_table(t1_results)
    print_task2_table(t2_results)
    print_feature_table()

    print("\n\nDone.")
    return {'task1': t1_results, 'task2': t2_results}


if __name__ == '__main__':
    results = main()
