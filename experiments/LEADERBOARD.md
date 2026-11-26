# Leaderboard

Evaluation version 2: 13 finished runs. All versions: 1712 runs recorded, 1065 failed, 634 finished under other versions (not comparable, not shown).
Skill = 1 - model pinball / best-baseline pinball on purged walk-forward validation folds.
Positive means better than the baseline. The final holdout is not used here.
Every run in this table was selected on the same validation folds, so the best score is
biased upward by the number of runs tried; only the final holdout settles it.

## Champions

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| nano | 20261126-104135-pc-5070ti-3409 | +0.0364 | +0.0111 | +0.0616 | 0.000 | 19,454 | 92 | 0.116 | 0.551 |
| mid | 20261126-115321-pc-5070ti-be9e | -0.0146 | -0.0696 | +0.0404 | 0.009 | 62,686 | 261 | 0.28 | 1.076 |
| large | 20261126-104349-pc-5070ti-1662 | +0.0059 | -0.0365 | +0.0482 | 0.600 | 8,588,318 | 33568 | 3.741 | 3.026 |

## Top nano runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| nano | 20261126-104135-pc-5070ti-3409 | +0.0364 | +0.0111 | +0.0616 | 0.000 | 19,454 | 92 | 0.116 | 0.551 |
| nano | 20261126-w5-nano-lookback256-s11 | +0.0270 | -0.0082 | +0.0622 | 0.029 | 19,070 | 91 | 0.171 | 0.988 |
| nano | 20261126-103739-pc-5070ti-fecd | +0.0106 | -0.0277 | +0.0489 | 0.857 | 19,070 | 91 | 0.229 | 1.089 |
| nano | 20261126-w1-nano-v1champ-s11 | +0.0088 | -0.0369 | +0.0546 | 0.898 | 19,070 | 91 | 0.155 | 0.54 |
| nano | 20261126-w2-nano-v1champ-s12 | -0.0075 | -0.0569 | +0.0419 | 0.296 | 19,070 | 91 | 0.157 | 0.54 |

## Top mid runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| mid | 20261126-115321-pc-5070ti-be9e | -0.0146 | -0.0696 | +0.0404 | 0.009 | 62,686 | 261 | 0.28 | 1.076 |
| mid | 20261126-w3-mid-v1champ-s11 | -0.0152 | -0.0697 | +0.0393 | 0.069 | 62,686 | 261 | 0.208 | 0.973 |
| mid | 20261126-w6-mid-wide-s11 | -0.0207 | -0.0784 | +0.0370 | 0.002 | 293,150 | 1165 | 0.767 | 1.842 |
| mid | 20261126-w4-mid-v1champ-s12 | -0.0208 | -0.0771 | +0.0355 | 0.003 | 62,686 | 261 | 0.214 | 0.973 |

## Top large runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| large | 20261126-104349-pc-5070ti-1662 | +0.0059 | -0.0365 | +0.0482 | 0.600 | 8,588,318 | 33568 | 3.741 | 3.026 |
| large | 20261126-112009-pc-5070ti-8e2e | +0.0007 | -0.0410 | +0.0423 | 0.231 | 8,514,590 | 33280 | 2.623 | 3.025 |
| large | 20261126-w7-large-regularised-s11 | -0.0052 | -0.0554 | +0.0451 | 0.071 | 2,190,878 | 8578 | 2.272 | 1.611 |
| large | 20261126-100329-pc-5070ti-763a | -0.0237 | -0.0923 | +0.0449 | 0.000 | 4,859,166 | 19001 | 3.193 | 2.756 |
