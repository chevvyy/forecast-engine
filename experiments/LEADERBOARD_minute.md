# Leaderboard

Evaluation version minute-2: 20 finished runs. All versions: 1858 runs recorded, 1065 failed, 773 finished under other versions (not comparable, not shown).
Skill = 1 - model pinball / best-baseline pinball on purged walk-forward validation folds.
Positive means better than the baseline. The final holdout is not used here.
Every run in this table was selected on the same validation folds, so the best score is
biased upward by the number of runs tried; only the final holdout settles it.

## Champions

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| nano | 20261127-a7-minute2-nano-s12 | +0.0266 | +0.0059 | +0.0473 | 0.000 | 9,256 | 49 | 0.049 | 0.122 |
| mid | 20261127-a8-minute2-mid-s11 | +0.0298 | +0.0083 | +0.0514 | 0.000 | 65,512 | 272 | 0.164 | 0.458 |
| large | 20261127-080923-pc-5070ti-2dea | +0.0197 | +0.0025 | +0.0370 | 0.011 | 2,156,072 | 8442 | 1.489 | 1.501 |

## Top nano runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| nano | 20261127-a7-minute2-nano-s12 | +0.0266 | +0.0059 | +0.0473 | 0.000 | 9,256 | 49 | 0.049 | 0.122 |
| nano | 20261127-065534-pc-5070ti-2679 | +0.0259 | +0.0043 | +0.0476 | 0.000 | 9,256 | 49 | 0.067 | 0.122 |
| nano | 20261127-a6-minute2-nano-s11 | +0.0256 | +0.0029 | +0.0483 | 0.000 | 9,256 | 49 | 0.066 | 0.122 |
| nano | 20261127-b6-minute2-nano-s13 | +0.0206 | -0.0006 | +0.0418 | 0.002 | 9,256 | 49 | 0.066 | 0.122 |
| nano | 20261127-081421-pc-5070ti-f72b | +0.0176 | +0.0008 | +0.0344 | 0.003 | 11,272 | 57 | 0.094 | 0.122 |

## Top mid runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| mid | 20261127-a8-minute2-mid-s11 | +0.0298 | +0.0083 | +0.0514 | 0.000 | 65,512 | 272 | 0.164 | 0.458 |
| mid | 20261127-b8-minute2-mid-s13 | +0.0274 | +0.0062 | +0.0486 | 0.000 | 65,512 | 272 | 0.142 | 0.458 |
| mid | 20261127-074307-pc-5070ti-89b9 | +0.0268 | +0.0074 | +0.0461 | 0.000 | 61,672 | 257 | 0.13 | 0.461 |
| mid | 20261127-062114-pc-5070ti-2b16 | +0.0256 | +0.0073 | +0.0438 | 0.000 | 269,608 | 1069 | 0.349 | 0.926 |
| mid | 20261127-074207-pc-5070ti-4328 | +0.0246 | +0.0029 | +0.0462 | 0.000 | 65,512 | 272 | 0.194 | 0.458 |
| mid | 20261127-081342-pc-5070ti-1885 | +0.0232 | +0.0064 | +0.0400 | 0.000 | 61,672 | 257 | 0.086 | 0.247 |
| mid | 20261127-081245-pc-5070ti-88ab | +0.0224 | +0.0036 | +0.0413 | 0.000 | 115,048 | 466 | 0.238 | 0.689 |
| mid | 20261127-b7-minute2-mid-s12 | +0.0218 | -0.0051 | +0.0487 | 0.070 | 65,512 | 272 | 0.193 | 0.458 |
| mid | 20261127-060827-pc-5070ti-5f02 | +0.0189 | +0.0060 | +0.0317 | 0.000 | 65,512 | 272 | 0.145 | 0.458 |
| mid | 20261127-074345-pc-5070ti-edb7 | +0.0181 | -0.0019 | +0.0380 | 0.008 | 65,512 | 272 | 0.193 | 0.458 |

## Top large runs

| Tier | Run | Score | Return skill | Vol skill | DM p (all) | Params | ONNX KB | CPU ms | Peak VRAM GB |
|---|---|---|---|---|---|---|---|---|---|
| large | 20261127-080923-pc-5070ti-2dea | +0.0197 | +0.0025 | +0.0370 | 0.011 | 2,156,072 | 8442 | 1.489 | 1.501 |
| large | 20261127-074648-pc-5070ti-b723 | +0.0113 | -0.0129 | +0.0355 | 0.822 | 2,199,080 | 8610 | 2.158 | 1.502 |
| large | 20261127-052822-pc-5070ti-f800 | +0.0001 | -0.0069 | +0.0071 | 0.507 | 2,199,080 | 8610 | 2.209 | 1.502 |
| large | 20261127-060942-pc-5070ti-f383 | -0.0469 | -0.0530 | -0.0409 | 0.000 | 8,592,424 | 33584 | 6.139 | 3.996 |
