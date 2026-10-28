# Evaluation Protocol

If this document and a result disagree, the result is wrong until proven otherwise.

## 1. Leakage rules (non-negotiable)

1. **Point-in-time features.** Every feature at time `t` uses only data with timestamp `< t` (bar close known).
   Rolling windows are right-closed at `t-1`. Normalization statistics come from the training fold only.
2. **Labels record their interval.** Each sample stores `(t_start, t_end)` of the information its label uses
   (e.g. a 30-min forward return spans `t` → `t+30`). Purging is impossible without this.
3. **No shuffling across time.** Ever. Random k-fold on overlapping labels inflates scores.
4. **Hyperparameters are tuned inside an inner split**, never on the outer test fold.
5. **One final holdout** (the most recent ~6 months) is touched exactly once, at the end of a phase.

## 2. Splitting scheme

**Primary: expanding walk-forward with purge + embargo.**

```
|---- train ----|purge|-- val --|embargo|
|------- train -------|purge|-- val --|embargo|
|----------- train -----------|purge|-- val --|embargo|
                                                 ...           |== final holdout ==|
```

- Purge: drop training samples whose label interval overlaps the validation interval.
- Embargo: drop a buffer after each validation block (≥ the longest label horizon, plus feature lookback
  if features use future-adjacent aggregates).
- Walk-forward answers the deployment question ("trained on the past, how does it do next?"). Purged k-fold
  can train on data after the test fold, so it estimates a stable-regime average instead.

**Secondary (Phase 3+): Combinatorial Purged CV (CPCV)** to get a *distribution* of backtest paths instead of
one, which feeds the Deflated Sharpe Ratio.

## 3. Metrics

| Category | Metric | Notes |
|---|---|---|
| Point accuracy | MAE, RMSE **relative to the zero-return / random-walk baseline** | Absolute RMSE on returns is meaningless alone. |
| Probabilistic | **Pinball loss per quantile, CRPS**, coverage of 80% interval | Primary metric for quantile models. |
| Volatility | **QLIKE**, MSE on log RV, vs **HAR-RV** | QLIKE is robust to noisy RV proxies. |
| Ranking | **IC / RankIC** (Spearman of predicted vs realized cross-sectional returns), IC IR | Per-timestamp, then averaged. |
| Direction | Hit rate **with a binomial test vs 50%** and vs the always-up baseline | 52% on 10k samples may be noise; compute it. |
| Significance | **Diebold–Mariano test** vs the baseline, HAC-adjusted | Required for any "beats baseline" claim. |
| Stability | Metric per fold, per year, per regime (high/low VIX) | A model that only wins in 2020 doesn't win. |

## 4. Backtest (only after metrics pass)

- Simple, transparent strategy: e.g. go long top-k / short bottom-k by predicted return each rebalance, or
  size by predicted return / predicted vol.
- **Costs always on:** commission (Alpaca: $0) + spread (half-spread per side; estimate from quotes or assume
  ≥1–2 bps for megacaps) + slippage. Report gross and net.
- Report: net annualized return, volatility, Sharpe, max drawdown, turnover, **Deflated Sharpe Ratio**
  (corrects for how many configurations you tried — keep a count in `experiments/LOG.md`).
- Execution assumption: signal at bar close `t`, fill at open of `t+1` or worse. Never fill at the close
  that generated the signal.

## 5. Mandatory baselines (logged in every comparison table)

1. Zero return
2. Random walk (last return persists)
3. Rolling mean return
4. HAR-RV (volatility)
5. Previous best model in the ladder

## 6. Result card template

Every reportable run produces a card in `experiments/cards/` — see `experiments/README.md`.

## Sources

- Purged cross-validation overview — https://en.wikipedia.org/wiki/Purged_cross-validation
- Purge/embargo implementation notes and walk-forward vs purged k-fold — https://www.quantt.co.uk/resources/purged-cross-validation
- `purgedcv` library (reference implementation to cross-check against) — https://github.com/eslazarev/purged-cross-validation
- López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), ch. 7 and 12
