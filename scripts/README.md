# scripts

CLI entry points to write (thin wrappers over `src/forecast`). Planned commands:

| Command | Does |
|---|---|
| `fe-ingest --universe configs/data/universe.yaml` | Historical pull, resumable |
| `fe-dq` | Data-quality report for the store |
| `fe-features --features configs/features/v1.yaml` | Build feature/label tables for a snapshot |
| `fe-baselines` | Run rung 0–1 on all folds, write result cards |
| `fe-study create --model patchtst` | Create an Optuna study from a model's search space |
| `fe-worker --study <name>` | Start a GPU worker (run one per machine) |
| `fe-eval --run <mlflow_run_id>` | Full evaluation + result card |
| `fe-serve` | Start the prediction API |

Register them under `[project.scripts]` in `pyproject.toml` when implemented.
