# docker

Planned images:

| Image | Base | Purpose |
|---|---|---|
| `forecast-train` | `pytorch/pytorch:*-cuda12.8-cudnn9-runtime` (or newer) | GPU worker; run with `--gpus all` on either machine |
| `forecast-serve` | `python:3.12-slim` | CPU inference API for k3s (forecast-chat namespace) |
| `forecast-ingest` | `python:3.12-slim` | Historical + live ingestion jobs |

Notes:
- Blackwell needs CUDA 12.8+ images; older PyTorch images lack sm_120 kernels.
- Mount `DATA_ROOT` (NFS from TrueNAS) and a local NVMe cache volume into `forecast-train`.
- Shared services (Postgres for Optuna + MLflow, MLflow server) are deployed as k3s manifests — add them
  under `docker/k8s/` when written.
