# Roadmap

Each phase ends with an exit criterion. Don't start the next phase until it's met.

## Phase 0 — Environment (≈ 1 evening)
- [ ] `uv` installed on both GPU machines; `uv sync` succeeds
- [ ] `torch.cuda.is_available()` and correct arch list on both cards (sm_120 on the 5070 Ti)
- [ ] Postgres + MLflow running on k3s; both machines can reach them
- [ ] Alpaca paper account + keys in `.env`
**Exit:** a dummy MLflow run logged from each machine.

## Phase 1 — Data layer (≈ 1–2 weeks)
- [ ] Resumable historical ingestor (daily + 1-min, SIP, raw) for the initial universe, 2016→now
- [ ] Corporate-actions table stored; adjustment as a pure derived step
- [ ] Exchange calendar + regular-session minute grid with `is_filled`
- [ ] Data-quality report: missing minutes, OHLC sanity (low ≤ open/close ≤ high), split checks vs second source
**Exit:** all `tests/data` pass; DQ report clean for the universe.

## Phase 2 — Features, labels, splits, baselines (≈ 1–2 weeks)
- [ ] Feature builders with declared lookbacks; look-ahead property tests
- [ ] Labels: forward log return and realized vol at 5/15/30/60 min, with label intervals
- [ ] Purged walk-forward splitter, cross-checked against `purgedcv`
- [ ] Rung 0–1 baselines + metrics + Diebold–Mariano
**Exit:** baseline result cards for every horizon. HAR-RV numbers look sane.

## Phase 3 — GBDT and compact deep models (≈ 2–4 weeks)
- [ ] LightGBM (GPU) global model; Optuna study shared by both workers
- [ ] PatchTST and ModernTCN with RevIN, patching, quantile heads, bf16
- [ ] Ablations: lookback length, patch size, feature groups, global vs per-sector
**Exit:** a model beats the best baseline with DM p < 0.05 on validation folds for at least the vol target.

## Phase 4 — Foundation models (≈ 2 weeks)
- [ ] TiRex and Kronos zero-shot on the same folds
- [ ] Frozen embeddings → LightGBM
- [ ] Fine-tune the better one on the 5070 Ti
**Exit:** documented win, or documented "no gain" with numbers.

## Phase 5 — Live minute pipeline (≈ 2 weeks)
- [ ] Websocket ingestor (IEX for plumbing; SIP when evaluating for real)
- [ ] Incremental features; latency budget < 5 s from bar close to prediction
- [ ] Serving API consumed by forecast-chat
- [ ] Paper-trading-style shadow log: predictions vs realized, monitored daily for drift
**Exit:** one month of live shadow predictions whose metrics match the backtest within tolerance.

## Phase 6 — Decide
Touch the final holdout once. Write up results (including null results). Then decide whether a trading layer
is worth building at all.
