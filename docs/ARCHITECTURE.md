# Architecture

## 1. Pipeline

```
 Alpaca REST (history)          Alpaca websocket (live, Phase 5)
        │                                  │
        ▼                                  ▼
 data/ingest ──► raw Parquet (NAS) ◄── data/live_ingest
        │
        ▼
 data/adjust + data/calendar ──► adjusted, session-aligned bars
        │
        ▼
 features/build ──► feature matrix + labels (with label intervals)
        │
        ▼
 evaluation/splits ──► purged walk-forward folds
        │
        ▼
 training/worker ◄──── Optuna study (Postgres) ────► MLflow (runs, artifacts)
        │
        ▼
 models/{baselines, gbdt, deep, foundation}
        │
        ▼
 evaluation/{metrics, stats, backtest} ──► result cards
        │
        ▼
 serving/ ──► predictions table / HTTP API ──► forecast-chat
```

## 2. Module contracts

Each contract is the interface to implement. Types are described, not coded.

### data
- `ingest(symbols, start, end, timeframe) → writes raw partitions`, idempotent, resumable via manifest.
- `adjust(raw_bars, corporate_actions) → adjusted_bars` (pure function; raw never mutated).
- `session_align(bars, calendar, session="regular") → bars with full minute grid + is_filled flag`.

### features
- `build(bars, feature_config) → (X, y, meta)` where `meta` has `security_id, t, label_start, label_end`.
- Every feature declares its **lookback** and is tested for look-ahead.
- Feature groups (configurable): returns at multiple lags, realized vol windows, range-based vol
  (Parkinson / Garman–Klass), volume z-scores, VWAP distance, time-of-day / day-of-week encodings,
  market & sector ETF returns, cross-sectional ranks.

### evaluation
- `splits(meta, scheme, horizon, embargo) → iterator of (train_idx, val_idx)`.
- `metrics(y_true, y_pred_quantiles, baseline) → dict`.
- `diebold_mariano(loss_a, loss_b, h) → (stat, p)`.
- `backtest(predictions, prices, cost_model) → equity curve + stats`.

### models
All models implement one interface: `fit(train, val, config)`, `predict(X) → quantiles[n, horizons, q]`,
`save(path)`, `load(path)`, `vram_estimate(config)`. Baselines, LightGBM, PatchTST, TiRex adapters all comply,
so evaluation and serving never special-case a model.

### training
- `worker(study_name)`: loop — ask Optuna for a trial, check VRAM fits, run all folds, report, repeat.
- Checkpoint every N steps to NAS; resumable.
- Seeds fixed and logged; data snapshot hash logged with every run.

### serving
- Batch mode: after market close, score next-day/next-session; write to a predictions table.
- Live mode (Phase 5): on each new minute bar, update features incrementally and score. Recurrent models
  (TiRex-2 style) suit this because state updates cost the same per step.
- Exposes a small read-only HTTP API (FastAPI) for forecast-chat: latest quantiles + vol for a symbol, plus
  the model version and its validation metrics so the chat can cite them honestly.

## 3. Reproducibility

A run is reproducible from: git commit + config hash + data snapshot hash + seed. All four are logged to MLflow.

## 4. Deployment shape

- Training workers: processes/containers on the GPU machines.
- k3s: Postgres (Optuna + MLflow backend), MLflow server, serving API in the existing forecast-chat namespace.
- NAS: Parquet store + MLflow artifacts.
