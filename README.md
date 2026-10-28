# forecast-engine

A from-scratch forecasting engine for US equities: ingest minute bars, build leak-free features, train
models on whichever GPU is free (RTX 5070 Ti 16 GB or RTX 3060 12 GB), and serve predictions to the
forecast-chat demo.

> Status: **scaffold only.** No implementation code yet. Every module folder contains a `README.md`
> describing what goes there, its inputs/outputs, and the tests that must exist before it counts as done.

---

## The short version of the research

Read `docs/RESEARCH.md` for the full argument and sources. The decisions it lands on:

| Question | Decision | Why (one line) |
|---|---|---|
| What do we predict? | **Log returns and realized volatility over fixed horizons**, as quantiles — never raw price | Price is non-stationary; a model that "predicts price" mostly learns `y[t+1] ≈ y[t]`. |
| First model to beat | **Random walk / zero-return**, then **HAR-RV** for volatility | Published studies show many deep models and foundation models fail to beat these. |
| First real model | **LightGBM on engineered features** | Strongest, cheapest documented baseline for tabular financial features. |
| First deep model | **PatchTST-class / ModernTCN-class** compact model trained from scratch | Patching cuts sequence cost; small models fit 12 GB and train in minutes. |
| Foundation models? | **Kronos (finance-specific) and TiRex (xLSTM, 35M params) as challengers / feature extractors**, not as the plan | Zero-shot generic TSFMs (Chronos, TimesFM) underperform boosted trees on returns. |
| Language | **Python 3.12 + PyTorch ≥ 2.7 (CUDA 12.8+)**; Polars + Parquet + DuckDB for data | Every relevant model ships in PyTorch; Blackwell (5070 Ti) needs cu128+ wheels. |
| Multi-GPU | **Independent trials per GPU via a shared Optuna study**, not DDP across machines | DDP waits on the slowest card and needs fast interconnect; trials don't. |
| Validation | **Walk-forward with purge + embargo**, Diebold–Mariano vs. baseline, costs in every backtest | Leakage is the #1 reason finance models look great in research and fail live. |
| Data (phase 1) | **Alpaca free tier** historical minute bars (SIP history, `adjustment=all`) | Free, minute resolution, history back to 2016. Live websocket comes later. |

---

## Repository layout

```
forecast-engine/
├── README.md                 ← you are here
├── CLAUDE.md                 ← conventions for AI coding agents working in this repo
├── pyproject.toml            ← dependency declaration (no code)
├── .env.example              ← API keys and paths; copy to .env
├── docs/
│   ├── RESEARCH.md           ← model survey, evidence, final recommendation
│   ├── LANGUAGES_AND_STACK.md← why Python/PyTorch/Polars, what was rejected
│   ├── DATA_SOURCES.md       ← minute-data providers, pricing, data pitfalls
│   ├── EVALUATION.md         ← leakage rules, CV scheme, metrics, backtest protocol
│   ├── COMPUTE.md            ← two-GPU scheduling, VRAM budgets, Blackwell notes
│   ├── ARCHITECTURE.md       ← pipeline and module contracts
│   ├── ROADMAP.md            ← phased plan with exit criteria
│   └── adr/                  ← architecture decision records (one decision per file)
├── configs/                  ← YAML configs (data, features, model, train, eval)
├── src/forecast/
│   ├── data/                 ← ingestion, adjustment, calendar, storage
│   ├── features/             ← point-in-time feature builders + labels
│   ├── models/               ← baselines, GBDT, deep models, foundation-model adapters
│   ├── training/             ← trainer, HPO worker, checkpointing
│   ├── evaluation/           ← splitters, metrics, backtester, stat tests
│   ├── serving/              ← batch/live inference, API for forecast-chat
│   └── utils/                ← logging, seeding, device selection
├── scripts/                  ← CLI entry points (documented, not written)
├── experiments/              ← experiment log + result cards
├── notebooks/                ← exploration only; nothing here is imported
├── tests/                    ← test plan; leakage tests are mandatory
├── data/                     ← git-ignored; local cache (real store is the NAS)
├── docker/                   ← training/serving image notes
└── .github/workflows/        ← CI skeleton
```

## Where to start

1. Read `docs/ROADMAP.md` → Phase 0 and Phase 1.
2. Copy `.env.example` to `.env` and add Alpaca paper-account keys.
3. Implement `src/forecast/data/` first, following its README. Nothing else is worth building until the
   data layer passes its tests (adjustments, calendar, gap handling).
4. Log every run in `experiments/LOG.md` using the template there.

## Non-goals (for now)

- Live trading / order execution. Predictions only; trading is a separate, later decision.
- Tick or order-book data. Minute bars first; L2 data is a different (expensive) project.
- Sub-minute latency. At one-minute cadence, Python is fast enough end to end.

## Disclaimer

This is a research and learning project. Nothing it produces is investment advice.
