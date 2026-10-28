# Languages and Stack

The question was not "Python because everyone uses it" — it was "what minimizes time-to-correct-result and
compute per experiment on two consumer GPUs." Each layer was evaluated separately.

## 1. Modeling language / framework

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **Python + PyTorch** | Every model in RESEARCH.md (PatchTST, ModernTCN, TiRex, Kronos, Chronos, TimesFM) ships PyTorch weights/code. `torch.compile`, bf16, CUDA 12.8+ Blackwell wheels. | Python overhead in data loading if done naively. | **Chosen.** |
| Python + JAX | Fast XLA compilation, clean functional style. | Few TS models ported; you'd re-implement everything. Gains show mostly on TPUs. | Rejected for now. |
| Julia (Flux/Lux) | Fast numerics, nice for research. | Tiny TS/finance DL ecosystem; harder to hire/learn alongside. | Rejected. |
| Rust (burn/candle) | Fast, safe, great for services. | Training ecosystem immature; no pretrained TS models. | Rejected for training; **candidate for the live ingestor** later. |
| C++ (libtorch) | Max inference speed. | Development cost; zero benefit at minute cadence. | Rejected. |

**Why speed isn't a reason to leave Python here:** one-minute bars give ~390 regular-session bars per symbol
per day. Even 500 symbols is ~200k rows/day — trivial. The bottleneck is GPU matmuls (already C++/CUDA) and
feature engineering (solved by Polars, below), not the interpreter.

**Blackwell note:** the RTX 5070 Ti is compute capability sm_120. It requires **PyTorch ≥ 2.7 built against
CUDA 12.8+**. Older wheels (cu124/cu126) fail with "no kernel image is available." Pin the cu128 (or newer)
index in `pyproject.toml` and verify with `torch.cuda.get_arch_list()` containing `sm_120`. The 3060 (sm_86)
works with the same wheels.

## 2. Data processing

| Option | Verdict | Reason |
|---|---|---|
| pandas | Use only at library boundaries | Single-threaded eager execution; 2026 benchmarks show Polars ~5–9× faster on group-bys/joins and much lower memory on Parquet. |
| **Polars** | **Chosen** for feature engineering | Multi-core, lazy query planning, native Parquet, rolling/group-by-dynamic windows ideal for bars. |
| **DuckDB** | **Chosen** for ad-hoc SQL over the Parquet store | Embedded, zero-server, reads Parquet directly. Fits your SQL background. |
| Spark | Rejected | Cluster overhead for a dataset that fits on one machine. |

## 3. Storage

- **Parquet**, partitioned `symbol=XXX/year=YYYY/month=MM/`, zstd compression, on the TrueNAS share.
  Columnar, compresses OHLCV ~5–10×, readable by Polars/DuckDB/PyArrow.
- **Local NVMe cache** on each training box for the active dataset (NFS latency hurts random window reads).
- Live phase: optional **TimescaleDB or QuestDB** on k3s for the rolling recent window; Parquet stays the
  source of truth for training.
- Note: the NAS pool has had reliability problems — keep raw downloads re-fetchable (store request manifests)
  and back up anything derived that is expensive to rebuild.

## 4. ML tooling

| Need | Choice | Alternative considered |
|---|---|---|
| Configs | **Hydra** (or plain YAML + pydantic) | gin, argparse |
| Experiment tracking | **MLflow**, self-hosted on k3s with Postgres + NAS artifact store | Weights & Biases (cloud), TensorBoard (too thin) |
| HPO / multi-GPU scheduling | **Optuna** with a shared Postgres storage | Ray Tune (more power, more moving parts — see COMPUTE.md) |
| GBDT | **LightGBM** (GPU build) + CatBoost | XGBoost (fine too) |
| Deep TS reference impls | **neuralforecast** (PatchTST, N-HiTS, iTransformer, TSMixer) for baselines; own `models/` for the main model | Time-Series-Library (THUML) |
| Foundation models | `tirex`, Kronos repo, `chronos-forecasting` | — |
| CV splitting | Own purged walk-forward implementation (cross-check against `purgedcv`) | `timeseriescv`, mlfinlab |
| Lint/format/type | ruff, mypy (strict on `src/`) | black+flake8 |
| Env / deps | **uv** | poetry, conda |
| Tests | pytest + hypothesis (property tests for leakage) | — |

## 5. Live data ingestion (later phase)

Python `asyncio` + the Alpaca websocket client is enough for minute bars. Revisit Rust only if you move to
trades/quotes streams for hundreds of symbols.

## Sources

- PyTorch Blackwell/sm_120 support (PyTorch forums) — https://discuss.pytorch.org/t/when-will-sm120-support-be-available/223621 ; https://discuss.pytorch.org/t/nvidia-geforce-rtx-5070-ti-with-cuda-capability-sm-120/221509
- Polars vs pandas vs DuckDB 2026 benchmark — https://uvik.net/blog/polars-vs-pandas/
- Polars vs pandas 3 on 10 GB — https://dev.to/johalputt/polars-10-vs-pandas-3-dataframe-performance-for-10gb-datasets-in-2026-3o8p
