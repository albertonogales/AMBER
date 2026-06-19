# Contributing to AMBER

Thank you for your interest in contributing! The guidelines below keep the codebase consistent and the review process smooth.

---

## Getting started

```bash
git clone https://github.com/albertonogales/AMBER.git
cd AMBER
pip install -e ".[dev]"   # installs runtime + dev dependencies
```

Run the test suite to make sure everything is green before you start:

```bash
pytest
```

---

## Reporting bugs

Open a GitHub Issue and include:

1. A minimal reproducible example.
2. The full traceback.
3. Your Python version and AMBER version (`python -c "import AMBER; print(AMBER.__version__)"`).

---

## Submitting changes

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Write your code.  Keep changes focused — one feature or fix per pull request.
3. Add or update tests in `tests/`.  The coverage must not drop below **99 %**.
4. Run the full suite before pushing:
   ```bash
   pytest --tb=short
   ```
5. Open a **Pull Request** against `main` with a clear description of what changed and why.

---

## Code style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Docstrings use the `:param` / `:return:` Sphinx style (see existing code).
- No inline comments unless the *why* is non-obvious.
- No emojis.

---

## Adding a new distance metric

1. Add the scalar function and the `*_distance_matrix` function in `AMBER/distances.py`.
2. Register both in `SIGNAL_DISTANCE_MATRIX` and add the name to `AVAILABLE_DISTANCES`.
3. Add tests in `tests/test_distances.py` following the existing pattern (identity, non-negativity, matrix shape, matrix-vs-scalar agreement).

---

## Adding a new normalisation strategy

1. Add a branch in `Map.__normalize` in `AMBER/map.py`.
2. Add tests in `tests/test_map.py` under `TestNormalization`.

---

## Contact

For questions, open a GitHub Discussion or e-mail alberto.nogales@uah.es.
