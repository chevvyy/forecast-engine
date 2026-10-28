# Compute: Training on Two GPUs "Depending on Availability"

## 1. Hardware

| GPU | Arch | VRAM | bf16 | Role |
|---|---|---|---|---|
| RTX 5070 Ti | Blackwell (sm_120) | 16 GB | Yes (also FP8) | Primary: larger deep models, foundation-model fine-tunes |
| RTX 3060 | Ampere (sm_86) | 12 GB | Yes | Secondary: GBDT, small deep models, HPO trials, inference |

The design does **not** assume which machine holds which card. Each worker announces its GPU and VRAM at
startup and only accepts jobs that fit (see `configs/train/hardware.yaml`).

## 2. Why not distributed data-parallel (DDP) across both machines

- DDP synchronizes gradients every step; the faster card idles waiting for the slower one.
- Cross-machine DDP needs fast interconnect. Gigabit Ethernet / Tailscale is far too slow for per-step
  all-reduce.
- The models in the ladder are small. A PatchTST run on a few years of minute bars fits in one GPU and
  finishes in minutes to an hour. There is nothing to parallelize *within* a run.

The real workload is **many independent runs** (HPO trials, walk-forward folds, ablations, seeds). That is
embarrassingly parallel, so parallelize *across* runs.

## 3. Chosen design: shared Optuna study as a job queue

```
           ┌──────────────── Postgres (k3s) ────────────────┐
           │  Optuna studies  +  MLflow tracking backend    │
           └──────────────▲───────────────────▲─────────────┘
                          │                   │
        ┌─────────────────┴──┐       ┌────────┴────────────┐
        │ worker @ 5070 Ti   │       │ worker @ 3060       │
        │ pulls next trial   │       │ pulls next trial    │
        │ writes metrics     │       │ writes metrics      │
        └─────────┬──────────┘       └─────────┬───────────┘
                  └──── NAS: Parquet data + MLflow artifacts ────┘
```

- Each machine runs `scripts/worker` (to be written) pointing at the same Optuna study name.
- Optuna's RDB storage handles concurrency; a worker that dies leaves its trial as FAIL and others continue.
- **Availability:** a worker simply isn't running when you're gaming or using the card. Start/stop it with a
  systemd unit or a k3s Job. No coordinator needs to know.
- **Pruning:** use Optuna's median or Hyperband pruner per fold so bad trials die early.
- **Walk-forward folds** are enumerated as separate Optuna trials or as a separate queue table — both workers
  can chew through folds of the same config.

### Alternative: Ray Tune
Ray handles heterogeneous GPUs across nodes and supports ASHA/PBT schedulers, plus LightGBM-Ray/XGBoost-Ray.
It's more powerful and more moving parts (head node, object store, version-matched Python on every node).
Adopt it only if Optuna-as-queue becomes a bottleneck (ADR-0004).

## 4. Where workers run

Existing direction: training on the personal GPU, keeping k3s simple. This design respects that:
- **Workers** run as plain processes (or Docker containers with `--gpus all`) on whichever machine has the GPU.
- **Shared services** (Postgres, MLflow UI) run on k3s, reached over Tailscale or `*.lan` hostnames.
- If the cards later move into k3s via the NVIDIA device plugin, the worker becomes a k3s Job with a GPU
  resource request — no code change, just a manifest.

## 5. VRAM budgets (planning numbers, verify empirically)

| Workload | Approx. VRAM (bf16) | Fits 3060? | Fits 5070 Ti? |
|---|---|---|---|
| LightGBM GPU, 50M rows × 200 features | 2–6 GB | Yes | Yes |
| PatchTST ~1–5M params, lookback 512, batch 256 | 2–6 GB | Yes | Yes |
| ModernTCN similar size | 2–6 GB | Yes | Yes |
| TiRex 35M inference / light fine-tune | 2–8 GB | Yes | Yes |
| Kronos base/large fine-tune | 8–16+ GB | Small only | Base yes; large may need grad checkpointing |

Rules: pre-materialize windows; pin memory; keep the active dataset on local NVMe; log
`torch.cuda.max_memory_allocated()` per run so the scheduler's VRAM requirements are data-driven.

## 6. Software prerequisites per machine

- NVIDIA driver supporting CUDA 12.8+ (needed for Blackwell)
- `uv sync` from this repo (pins the cu128+ PyTorch index)
- Verify: `torch.cuda.get_arch_list()` includes `sm_120` on the 5070 Ti box
- Same Python minor version on both machines (pickled Optuna/MLflow artifacts are more portable that way)

## Sources

- Ray Tune / LightGBM-Ray multi-node GPU notes — https://github.com/ray-project/ray/blob/ray-1.12.1/doc/source/ray-more-libs/lightgbm-ray.rst ; https://insertchat.com/glossary/ray-tune
- PyTorch Blackwell support — https://discuss.pytorch.org/t/when-will-sm120-support-be-available/223621
