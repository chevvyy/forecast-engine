# ADR-0003: Purged, embargoed walk-forward as the primary validation scheme

- Status: Accepted
- Date: 2026-10-27

## Context
Forward-return labels overlap in time. Shuffled k-fold leaks future information. Purged k-fold can train on
data after its test fold, which estimates a stable-regime average rather than deployment performance.

## Decision
Expanding walk-forward folds with purge (by label interval) and embargo (≥ max horizon). CPCV added in Phase 3
for Deflated Sharpe. A final holdout (latest ~6 months) is used once per phase.

## Consequences
- Every sample must carry its label interval.
- Fewer effective training samples near fold boundaries; accepted.
- The number of configurations tried is counted and fed into the Deflated Sharpe Ratio.
