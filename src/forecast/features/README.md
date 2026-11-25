# forecast.features

## What exists
`build.py`: per-bar channels, lazy windows, labels and label intervals for daily bars, and for intraday
bars with `bar_seconds` set (a bar is known one bar after its start; no window or label crosses the
overnight gap or a missing bar; see the module docstring). The table below is the fuller plan.

## Files to create
| File | Responsibility |
|---|---|
| `registry.py` | Feature groups register a name, builder, and declared `lookback_min` |
| `returns.py`, `volatility.py`, `volume.py`, `calendar.py`, `market.py`, `cross_section.py` | Group builders (Polars lazy expressions) |
| `labels.py` | Forward log return + realized vol per horizon; emits `label_start`, `label_end` |
| `windows.py` | Materialize fixed-shape sequence windows for deep models into memory-mapped arrays |
| `normalize.py` | Fit stats on train fold only; apply to val/test |

## Rules
- Features at `t` use bars strictly before `t`'s close is known. Rolling windows end at `t-1` unless the
  feature is explicitly "as of bar close t" and the label starts at `t+1`.
- Cross-sectional ranks are computed per timestamp using only symbols with data at that timestamp.
- No global (full-sample) statistics anywhere.

## Required tests (`tests/features/`, marker `leakage`)
- **Future-perturbation test:** randomly change all data after time `t`; every feature at `≤ t` must be
  bit-identical. Run with hypothesis across random `t` and symbols.
- Label interval correctness for each horizon.
- Declared lookback ≥ actual lookback (perturb data before `t - lookback`; feature at `t` must not change).
