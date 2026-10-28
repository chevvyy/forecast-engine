# ADR-0004: Parallelize across runs with a shared Optuna study, not DDP

- Status: Accepted
- Date: 2026-10-27

## Context
Two heterogeneous consumer GPUs (RTX 5070 Ti 16 GB, RTX 3060 12 GB) on separate machines, available
intermittently. Models are small; the workload is many independent trials/folds.

## Decision
Each GPU runs an independent worker that pulls trials from a shared Optuna study (Postgres on k3s) and logs to
MLflow. Workers start/stop freely. No cross-machine gradient synchronization.

## Alternatives
- DDP across machines: rejected — network too slow; faster card waits on slower.
- Ray Tune: deferred — stronger scheduling, heavier ops. Revisit if the queue becomes a bottleneck or if
  population-based training is needed.

## Consequences
- Each worker must check VRAM before accepting a trial.
- Trials must be deterministic given (config, seed, data hash) so results from either card are comparable.
