# tests

Mirror `src/forecast/`: `tests/data`, `tests/features`, `tests/models`, `tests/training`, `tests/evaluation`.
Each module README lists its required tests.

Markers (defined in `pyproject.toml`):
- `leakage` — look-ahead guards. CI fails if any is skipped.
- `gpu` — needs CUDA; skipped in CI, run locally on both cards.
- `network` — hits Alpaca; skipped in CI.

The three most important tests in the repo:
1. Future-perturbation invariance of features (`tests/features`)
2. Purge/embargo correctness of the splitter (`tests/evaluation`)
3. Backtest fills strictly after signal time (`tests/evaluation`)
