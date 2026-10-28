# forecast.evaluation

Spec lives in `docs/EVALUATION.md`. This folder implements it.

## Files to create
| File | Responsibility |
|---|---|
| `splits.py` | Expanding walk-forward with purge (by label interval) + embargo; CPCV later |
| `metrics.py` | Pinball, CRPS (from quantiles), coverage, QLIKE, MAE/RMSE relative to baseline, IC/RankIC, hit rate + binomial p |
| `stats.py` | Diebold–Mariano with HAC variance; Deflated Sharpe Ratio |
| `backtest.py` | Signal → positions → net PnL with cost model; fills at next bar open |
| `report.py` | Render a result card (markdown) and log it to MLflow |

## Required tests (marker `leakage`)
- No train index has a label interval overlapping any val interval.
- Embargo removes the configured span after each val block.
- Splitter output matches `purgedcv` on a synthetic dataset.
- Backtest never fills on the bar that generated the signal.
- DM test reproduces a known textbook example.
