# Conventions for AI coding agents in this repo

- Read `docs/ARCHITECTURE.md` and the target module's `README.md` before writing code in it.
- Never introduce look-ahead: features at `t` use only data known before `t`. Any change to `features/`,
  `evaluation/splits.py`, or `evaluation/backtest.py` must keep all `leakage`-marked tests passing.
- Targets are returns and realized volatility as quantiles (ADR-0001). Do not add price-level targets.
- Polars for dataframes; pandas only at library boundaries. No naive datetimes (UTC everywhere).
- Every model implements the common interface in `models/base.py`.
- Every reportable result needs a result card and a baseline comparison with a DM test.
- Don't add dependencies without updating `pyproject.toml` and noting why in the PR description.
- Secrets come from `.env`; never hard-code keys.
