# forecast.models

All models implement the same interface (see `docs/ARCHITECTURE.md` §2):
`fit`, `predict → quantiles[n, horizons, q]`, `save`, `load`, `vram_estimate`.

## Files to create
| Path | Rung | Notes |
|---|---|---|
| `base.py` | — | Interface/protocol definition |
| `baselines/zero_return.py`, `random_walk.py`, `rolling_mean.py` | 0 | Quantiles from empirical residual distribution |
| `baselines/har_rv.py` | 0 | HAR-RV (daily/weekly/monthly RV components) for volatility |
| `linear.py` | 1 | Ridge / quantile regression |
| `gbdt.py` | 2 | LightGBM GPU, one booster per quantile (or multi-quantile wrapper) |
| `deep/layers.py` | 3 | RevIN, patch embedding, monotonic quantile head |
| `deep/patchtst.py` | 3 | Channel-independent patched Transformer |
| `deep/moderntcn.py` | 3 | Large-kernel depthwise conv blocks |
| `deep/global_embed.py` | 3 | Security + sector embeddings for one global model |
| `foundation/tirex_adapter.py` | 4 | Wrap TiRex to the common interface; embedding extraction |
| `foundation/kronos_adapter.py` | 4 | Wrap Kronos; OHLCV input |
| `ensemble.py` | 5 | Stacking on out-of-fold predictions only |

## Required tests
- Every model passes a shared contract test (shapes, monotonic quantiles, save/load round-trip).
- Deep models overfit a tiny synthetic batch to near-zero loss (sanity check the training loop).
- `vram_estimate` within 25% of measured `max_memory_allocated` on both GPUs.
