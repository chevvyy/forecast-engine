# forecast.training

## Files to create
| File | Responsibility |
|---|---|
| `trainer.py` | Train loop: bf16 autocast, grad clip, cosine schedule, early stopping on val CRPS, checkpointing |
| `worker.py` | Optuna worker loop: ask trial → VRAM check vs `hardware.yaml` → run all folds → report → repeat |
| `objective.py` | Maps a trial + config to fold runs and an aggregate score (mean val CRPS / QLIKE across folds) |
| `device.py` | Detect GPU, arch, free VRAM; choose profile from `WORKER_GPU_PROFILE` |
| `repro.py` | Seeds, deterministic flags, config hash, data snapshot hash, git commit → MLflow tags |

## Behavior
- A worker can be killed at any time. Trial heartbeats let Optuna mark stale trials failed.
- Checkpoints go to the NAS so a trial can resume on the other machine if needed.
- Every run logs: config, hashes, seed, GPU name, max VRAM, wall time, per-fold metrics.

## Required tests
- Two workers against a local SQLite Optuna study complete N trials without duplicates.
- Same (config, seed, data hash) on CPU twice → identical metrics (determinism check).
