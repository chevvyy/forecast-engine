# Leaderboard

Evaluation version minute-2: 43 finished runs. All versions: 1955 runs recorded, 1065 failed, 847 finished under other versions (not comparable, not shown).
Skill = 1 - model pinball / best-baseline pinball on purged walk-forward validation folds.
Positive means better than the baseline. The final holdout is not used here.
Every run in this table was selected on the same validation folds, so the best score is
biased upward by the number of runs tried; only the final holdout settles it.

## Champions

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| nano | 20261127-141843-pc-5070ti-b91f | +0.0328 | +0.0077 | +0.0579 | 0.000 | 9,256 | 49 | 0.065 | 0.159 |
| mid | 20261127-e5-minute2-mid-born-again-a8-s11 | +0.0312 | +0.0053 | +0.0571 | 0.000 | 65,512 | 272 | 0.197 | 0.487 |
| large | 20261127-160240-pc-5070ti-34b6 | +0.0244 | +0.0009 | +0.0480 | 0.002 | 2,156,072 | 8442 | 1.48 | 1.504 |

## Top nano runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| nano | 20261127-141843-pc-5070ti-b91f | +0.0328 | +0.0077 | +0.0579 | 0.000 | 9,256 | 49 | 0.065 | 0.159 |
| nano | 20261127-d8-minute2-distil-a8-nano-s11 | +0.0300 | +0.0053 | +0.0548 | 0.000 | 9,256 | 49 | 0.052 | 0.159 |
| nano | 20261127-151341-pc-5070ti-0af2 | +0.0298 | +0.0072 | +0.0523 | 0.000 | 11,920 | 63 | 0.09 | 0.159 |
| nano | 20261127-e4-minute2-distil-a8-nano-s12 | +0.0293 | +0.0065 | +0.0520 | 0.000 | 9,256 | 49 | 0.066 | 0.155 |
| nano | 20261127-f8-minute2-distil-e5-nano-s11 | +0.0285 | +0.0040 | +0.0530 | 0.000 | 9,256 | 49 | 0.068 | 0.156 |
| nano | 20261127-a7-minute2-nano-s12 | +0.0266 | +0.0059 | +0.0473 | 0.000 | 9,256 | 49 | 0.049 | 0.122 |
| nano | 20261127-065534-pc-5070ti-2679 | +0.0259 | +0.0043 | +0.0476 | 0.000 | 9,256 | 49 | 0.067 | 0.122 |
| nano | 20261127-a6-minute2-nano-s11 | +0.0256 | +0.0029 | +0.0483 | 0.000 | 9,256 | 49 | 0.066 | 0.122 |
| nano | 20261127-120939-pc-5070ti-d9db | +0.0234 | -0.0031 | +0.0500 | 0.002 | 9,256 | 49 | 0.067 | 0.12 |
| nano | 20261127-120859-pc-5070ti-b6ca | +0.0232 | +0.0050 | +0.0414 | 0.000 | 14,376 | 69 | 0.053 | 0.146 |

## Top mid runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| mid | 20261127-e5-minute2-mid-born-again-a8-s11 | +0.0312 | +0.0053 | +0.0571 | 0.000 | 65,512 | 272 | 0.197 | 0.487 |
| mid | 20261127-a8-minute2-mid-s11 | +0.0298 | +0.0083 | +0.0514 | 0.000 | 65,512 | 272 | 0.164 | 0.458 |
| mid | 20261127-c7-minute2-mid-batch2048-lr0003-s11 | +0.0281 | +0.0068 | +0.0494 | 0.000 | 65,512 | 272 | 0.175 | 0.579 |
| mid | 20261127-b8-minute2-mid-s13 | +0.0274 | +0.0062 | +0.0486 | 0.000 | 65,512 | 272 | 0.142 | 0.458 |
| mid | 20261127-074307-pc-5070ti-89b9 | +0.0268 | +0.0074 | +0.0461 | 0.000 | 61,672 | 257 | 0.13 | 0.461 |
| mid | 20261128-004249-pc-5070ti-c255 | +0.0267 | +0.0033 | +0.0501 | 0.000 | 65,512 | 272 | 0.143 | 0.486 |
| mid | 20261127-194021-pc-5070ti-1e82 | +0.0261 | -0.0025 | +0.0547 | 0.009 | 65,512 | 272 | 0.176 | 0.485 |
| mid | 20261127-062114-pc-5070ti-2b16 | +0.0256 | +0.0073 | +0.0438 | 0.000 | 269,608 | 1069 | 0.349 | 0.926 |
| mid | 20261127-105601-pc-5070ti-bfb0 | +0.0255 | +0.0019 | +0.0492 | 0.000 | 65,512 | 272 | 0.166 | 0.458 |
| mid | 20261127-150622-pc-5070ti-9500 | +0.0253 | +0.0029 | +0.0477 | 0.000 | 64,744 | 269 | 0.275 | 0.871 |

## Top large runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| large | 20261127-160240-pc-5070ti-34b6 | +0.0244 | +0.0009 | +0.0480 | 0.002 | 2,156,072 | 8442 | 1.48 | 1.504 |
| large | 20261127-080923-pc-5070ti-2dea | +0.0197 | +0.0025 | +0.0370 | 0.011 | 2,156,072 | 8442 | 1.489 | 1.501 |
| large | 20261127-102211-pc-5070ti-5790 | +0.0140 | -0.0017 | +0.0297 | 0.143 | 2,156,072 | 8442 | 0.897 | 1.324 |
| large | 20261127-180155-pc-5070ti-dada | +0.0113 | +0.0000 | +0.0225 | 0.033 | 2,156,072 | 8442 | 1.452 | 5.737 |
| large | 20261127-074648-pc-5070ti-b723 | +0.0113 | -0.0129 | +0.0355 | 0.822 | 2,199,080 | 8610 | 2.158 | 1.502 |
| large | 20261128-002040-pc-5070ti-c6d5 | +0.0108 | -0.0041 | +0.0256 | 0.402 | 2,156,072 | 8442 | 1.478 | 1.497 |
| large | 20261127-111456-pc-5070ti-3331 | +0.0026 | -0.0025 | +0.0077 | 0.982 | 3,233,320 | 12676 | 2.47 | 2.053 |
| large | 20261127-052822-pc-5070ti-f800 | +0.0001 | -0.0069 | +0.0071 | 0.507 | 2,199,080 | 8610 | 2.209 | 1.502 |
| large | 20261127-202719-pc-5070ti-a08c | -0.0399 | -0.0181 | -0.0617 | 0.000 | 2,156,072 | 8442 | 1.432 | 1.501 |
| large | 20261127-060942-pc-5070ti-f383 | -0.0469 | -0.0530 | -0.0409 | 0.000 | 8,592,424 | 33584 | 6.139 | 3.996 |

## Configurations by seed mean

One row per configuration (same shape, training settings, teacher / parent) that ran with at
least 2 different seeds, ranked by the mean over its seeds: the tables above rank single runs,
and the best single run is the luckiest seed. A score difference smaller than the sd column is
noise. Up to 10 per tier. Still selected on the validation folds, like everything here.

### nano

| Mean score | sd | Seeds | Min | Max | Mean return skill | Mean vol skill | Best run | How it was made |
|---|---|---|---|---|---|---|---|---|
| +0.0307 | 0.0018 | 3 | +0.0293 | +0.0328 | +0.0065 | +0.0549 | 20261127-141843-pc-5070ti-b91f | distilled from 20261127-a8-minute2-mid-s11 |
| +0.0241 | 0.0027 | 4 | +0.0206 | +0.0266 | +0.0013 | +0.0468 | 20261127-a7-minute2-nano-s12 | trained directly |

### mid

| Mean score | sd | Seeds | Min | Max | Mean return skill | Mean vol skill | Best run | How it was made |
|---|---|---|---|---|---|---|---|---|
| +0.0290 | 0.0032 | 2 | +0.0267 | +0.0312 | +0.0043 | +0.0536 | 20261127-e5-minute2-mid-born-again-a8-s11 | distilled from 20261127-a8-minute2-mid-s11 |
| +0.0263 | 0.0041 | 3 | +0.0218 | +0.0298 | +0.0031 | +0.0495 | 20261127-a8-minute2-mid-s11 | trained directly |
| +0.0186 | 0.0007 | 2 | +0.0181 | +0.0190 | -0.0012 | +0.0384 | 20261127-141652-pc-5070ti-5cac | prune 0.5 of 20261127-a8-minute2-mid-s11 |

### large

| Mean score | sd | Seeds | Min | Max | Mean return skill | Mean vol skill | Best run | How it was made |
|---|---|---|---|---|---|---|---|---|
| +0.0183 | 0.0069 | 3 | +0.0108 | +0.0244 | -0.0002 | +0.0369 | 20261127-160240-pc-5070ti-34b6 | trained directly |
| +0.0057 | 0.0079 | 2 | +0.0001 | +0.0113 | -0.0099 | +0.0213 | 20261127-074648-pc-5070ti-b723 | trained directly |
