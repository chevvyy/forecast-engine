# Leaderboard

Evaluation version 2: 194 finished runs. All versions: 1934 runs recorded, 1065 failed, 675 finished under other versions (not comparable, not shown).
Skill = 1 - model pinball / best-baseline pinball on purged walk-forward validation folds.
Positive means better than the baseline. The final holdout is not used here.
Every run in this table was selected on the same validation folds, so the best score is
biased upward by the number of runs tried; only the final holdout settles it.

## Champions

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| nano | 20261127-160602-pc-5070ti-5167 | +0.0480 | +0.0193 | +0.0767 | 0.000 | 19,454 | 92 | 0.158 | 0.651 |
| mid | 20261127-e1-mid-born-again-k15-s11 | +0.0452 | +0.0148 | +0.0756 | 0.000 | 61,534 | 256 | 0.246 | 1.269 |
| large | 20261127-180527-pc-5070ti-161c | +0.0436 | +0.0137 | +0.0736 | 0.000 | 2,450,718 | 9593 | 1.674 | 6.339 |

## Top nano runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| nano | 20261127-160602-pc-5070ti-5167 | +0.0480 | +0.0193 | +0.0767 | 0.000 | 19,454 | 92 | 0.158 | 0.651 |
| nano | 20261127-d3-distil-b1-nano-batch2048-s11 | +0.0479 | +0.0189 | +0.0770 | 0.000 | 19,454 | 92 | 0.159 | 0.725 |
| nano | 20261127-c1-distil-b1-nano-s11 | +0.0478 | +0.0197 | +0.0760 | 0.000 | 19,454 | 92 | 0.142 | 0.653 |
| nano | 20261127-c2-distil-b1-nano-s12 | +0.0472 | +0.0183 | +0.0760 | 0.000 | 19,454 | 92 | 0.122 | 0.652 |
| nano | 20261127-154604-pc-5070ti-5644 | +0.0467 | +0.0168 | +0.0765 | 0.000 | 19,454 | 92 | 0.157 | 0.651 |
| nano | 20261126-z6-nano-lr0003-s13 | +0.0466 | +0.0217 | +0.0715 | 0.000 | 19,454 | 92 | 0.158 | 0.553 |
| nano | 20261127-154235-pc-5070ti-5a03 | +0.0456 | +0.0173 | +0.0739 | 0.000 | 13,214 | 68 | 0.118 | 0.528 |
| nano | 20261127-c3-distil-b1-nano-lr0003-s11 | +0.0455 | +0.0149 | +0.0762 | 0.000 | 19,454 | 92 | 0.136 | 0.652 |
| nano | 20261127-195558-pc-5070ti-7819 | +0.0452 | +0.0165 | +0.0740 | 0.000 | 6,670 | 42 | 0.079 | 0.435 |
| nano | 20261127-073816-pc-5070ti-03cf | +0.0451 | +0.0147 | +0.0755 | 0.000 | 19,454 | 92 | 0.112 | 0.651 |

## Top mid runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| mid | 20261127-e1-mid-born-again-k15-s11 | +0.0452 | +0.0148 | +0.0756 | 0.000 | 61,534 | 256 | 0.246 | 1.269 |
| mid | 20261127-155512-pc-5070ti-c1f5 | +0.0442 | +0.0146 | +0.0739 | 0.000 | 61,534 | 256 | 0.242 | 1.27 |
| mid | 20261127-193302-pc-5070ti-b86c | +0.0437 | +0.0142 | +0.0732 | 0.000 | 61,534 | 256 | 0.233 | 1.113 |
| mid | 20261127-e2-mid-born-again-k15-s12 | +0.0416 | +0.0128 | +0.0704 | 0.000 | 61,534 | 256 | 0.232 | 1.269 |
| mid | 20261127-e3-mid-born-again-b1-s12 | +0.0410 | +0.0122 | +0.0698 | 0.000 | 60,382 | 252 | 0.168 | 1.269 |
| mid | 20261127-d4-mid-born-again-b1-s11 | +0.0400 | +0.0110 | +0.0691 | 0.000 | 60,382 | 252 | 0.225 | 1.27 |
| mid | 20261127-b1-mid-k9-batch2048-lr0003-s12 | +0.0387 | +0.0115 | +0.0659 | 0.000 | 60,382 | 252 | 0.213 | 1.171 |
| mid | 20261127-b2-mid-k9-batch2048-lr0003-s13 | +0.0383 | +0.0099 | +0.0667 | 0.000 | 60,382 | 252 | 0.216 | 1.171 |
| mid | 20261127-c5-mid-d96-batch2048-lr0003-s11 | +0.0374 | +0.0102 | +0.0645 | 0.000 | 127,422 | 514 | 0.263 | 1.748 |
| mid | 20261127-a3-mid-k9-batch2048-lr0003-s11 | +0.0356 | +0.0068 | +0.0643 | 0.000 | 60,382 | 252 | 0.188 | 1.171 |

## Top large runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| large | 20261127-180527-pc-5070ti-161c | +0.0436 | +0.0137 | +0.0736 | 0.000 | 2,450,718 | 9593 | 1.674 | 6.339 |
| large | 20261127-183754-pc-5070ti-6f42 | +0.0434 | +0.0143 | +0.0724 | 0.000 | 4,842,270 | 18948 | 3.363 | 9.895 |
| large | 20261127-e6-large-c6-distil-b1-s11 | +0.0433 | +0.0142 | +0.0725 | 0.000 | 2,450,718 | 9593 | 1.582 | 6.339 |
| large | 20261127-173032-pc-5070ti-cbfc | +0.0423 | +0.0111 | +0.0734 | 0.000 | 2,506,014 | 9809 | 2.639 | 6.34 |
| large | 20261127-194124-pc-5070ti-aff8 | +0.0412 | +0.0122 | +0.0702 | 0.000 | 1,254,174 | 4919 | 1.002 | 4.173 |
| large | 20261127-c6-large-01c4-batch2048-lr0003-s11 | +0.0243 | -0.0095 | +0.0581 | 0.009 | 2,450,718 | 9593 | 1.76 | 6.239 |
| large | 20261127-d2-large-c6-recipe-s13 | +0.0213 | -0.0131 | +0.0557 | 0.039 | 2,450,718 | 9593 | 1.738 | 6.239 |
| large | 20261127-d1-large-c6-recipe-s12 | +0.0174 | -0.0233 | +0.0581 | 0.276 | 2,450,718 | 9593 | 1.73 | 6.238 |
| large | 20261127-145020-pc-5070ti-7cfd | +0.0169 | -0.0199 | +0.0538 | 0.210 | 2,450,718 | 9593 | 1.767 | 6.239 |
| large | 20261126-233941-pc-5070ti-ed9e | +0.0166 | -0.0005 | +0.0336 | 0.010 | 4,316,190 | 16880 | 2.988 | 2.464 |

## Configurations by seed mean

One row per configuration (same shape, training settings, teacher / parent) that ran with at
least 2 different seeds, ranked by the mean over its seeds: the tables above rank single runs,
and the best single run is the luckiest seed. A score difference smaller than the sd column is
noise. Up to 10 per tier. Still selected on the validation folds, like everything here.

### nano

| Mean score | sd | Seeds | Min | Max | Mean return skill | Mean vol skill | Best run | How it was made |
|---|---|---|---|---|---|---|---|---|
| +0.0467 | 0.0014 | 3 | +0.0451 | +0.0478 | +0.0176 | +0.0758 | 20261127-c1-distil-b1-nano-s11 | distilled from 20261127-b1-mid-k9-batch2048-lr0003-s12 |
| +0.0431 | 0.0038 | 3 | +0.0391 | +0.0466 | +0.0164 | +0.0697 | 20261126-z6-nano-lr0003-s13 | trained directly |
| +0.0404 | 0.0004 | 2 | +0.0402 | +0.0407 | +0.0126 | +0.0683 | 20261126-233005-pc-5070ti-3afe | prune 0.3 of 20261126-z6-nano-lr0003-s13 |
| +0.0394 | 0.0037 | 3 | +0.0365 | +0.0436 | +0.0113 | +0.0675 | 20261126-x4-nano-champ-batch2048-s11 | trained directly |
| +0.0323 | 0.0046 | 6 | +0.0248 | +0.0364 | +0.0029 | +0.0617 | 20261126-x1-nano-champ-s11 | trained directly |
| +0.0007 | 0.0115 | 2 | -0.0075 | +0.0088 | -0.0469 | +0.0483 | 20261126-w1-nano-v1champ-s11 | trained directly |

### mid

| Mean score | sd | Seeds | Min | Max | Mean return skill | Mean vol skill | Best run | How it was made |
|---|---|---|---|---|---|---|---|---|
| +0.0437 | 0.0019 | 3 | +0.0416 | +0.0452 | +0.0141 | +0.0733 | 20261127-e1-mid-born-again-k15-s11 | distilled from 20261127-b1-mid-k9-batch2048-lr0003-s12 |
| +0.0405 | 0.0007 | 2 | +0.0400 | +0.0410 | +0.0116 | +0.0695 | 20261127-e3-mid-born-again-b1-s12 | distilled from 20261127-b1-mid-k9-batch2048-lr0003-s12 |
| +0.0375 | 0.0017 | 3 | +0.0356 | +0.0387 | +0.0094 | +0.0656 | 20261127-b1-mid-k9-batch2048-lr0003-s12 | trained directly |
| +0.0306 | 0.0035 | 3 | +0.0284 | +0.0346 | +0.0013 | +0.0598 | 20261127-a2-mid-k9-batch2048-s13 | trained directly |
| +0.0165 | 0.0032 | 4 | +0.0117 | +0.0185 | -0.0116 | +0.0445 | 20261126-y5-mid-champ-s12 | trained directly |
| +0.0151 | 0.0099 | 4 | +0.0009 | +0.0221 | -0.0208 | +0.0511 | 20261126-192624-pc-5070ti-066b | trained directly |
| +0.0137 | 0.0024 | 2 | +0.0120 | +0.0153 | -0.0281 | +0.0554 | 20261127-040002-pc-5070ti-913f | trained directly |
| +0.0046 | 0.0076 | 2 | -0.0007 | +0.0100 | -0.0269 | +0.0361 | 20261126-155436-pc-5070ti-9b79 | trained directly |
| -0.0180 | 0.0040 | 2 | -0.0208 | -0.0152 | -0.0734 | +0.0374 | 20261126-w3-mid-v1champ-s11 | trained directly |

### large

| Mean score | sd | Seeds | Min | Max | Mean return skill | Mean vol skill | Best run | How it was made |
|---|---|---|---|---|---|---|---|---|
| +0.0435 | 0.0002 | 2 | +0.0433 | +0.0436 | +0.0139 | +0.0730 | 20261127-180527-pc-5070ti-161c | distilled from 20261127-b1-mid-k9-batch2048-lr0003-s12 |
| +0.0200 | 0.0035 | 4 | +0.0169 | +0.0243 | -0.0165 | +0.0564 | 20261127-c6-large-01c4-batch2048-lr0003-s11 | trained directly |
| +0.0158 | 0.0011 | 2 | +0.0150 | +0.0166 | -0.0011 | +0.0327 | 20261126-233941-pc-5070ti-ed9e | prune 0.85 of 20261126-193730-pc-5070ti-a13a |
| +0.0137 | 0.0025 | 2 | +0.0120 | +0.0154 | -0.0242 | +0.0516 | 20261126-193730-pc-5070ti-a13a | trained directly |
| +0.0132 | 0.0014 | 3 | +0.0117 | +0.0144 | -0.0252 | +0.0516 | 20261126-190159-pc-5070ti-01c4 | trained directly |
| +0.0102 | 0.0019 | 3 | +0.0080 | +0.0117 | -0.0102 | +0.0305 | 20261127-105639-pc-5070ti-8264 | prune 0.85 of 20261127-c6-large-01c4-batch2048-lr0003-s11 |
| +0.0083 | 0.0017 | 2 | +0.0070 | +0.0095 | -0.0340 | +0.0506 | 20261126-151135-pc-5070ti-725a | trained directly |
| +0.0035 | 0.0033 | 2 | +0.0012 | +0.0059 | -0.0381 | +0.0451 | 20261126-104349-pc-5070ti-1662 | trained directly |
| -0.0035 | 0.0010 | 3 | -0.0046 | -0.0026 | -0.0553 | +0.0483 | 20261126-162920-pc-5070ti-7c2a | prune 0.5 of 20261126-151135-pc-5070ti-725a |
