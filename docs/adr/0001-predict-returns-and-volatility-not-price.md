# ADR-0001: Predict returns and realized volatility as quantiles, not price

- Status: Accepted
- Date: 2026-10-27

## Context
Price is non-stationary. Models trained on price levels learn persistence and report misleadingly low error.

## Decision
Targets are forward log returns and forward realized volatility at fixed horizons (5/15/30/60 min, 1 day),
predicted as quantiles with pinball loss. Price forecasts, if shown in forecast-chat, are derived from return
quantiles at display time.

## Consequences
- Baselines are zero-return / random walk / HAR-RV, and every metric is reported relative to them.
- Volatility gives a demonstrably working model early; return prediction stays an honest open question.
