# Planner runbook

The planner is a scheduled Claude session. It cannot reach the GPUs; it works only through this
branch (`loop`). Workers on the GPU machines pull the branch, run jobs, and push results. Each planner
cycle does the steps below in order, then pushes. Keep cycles small: one code improvement per cycle.

## Rules that never change
1. The final holdout (`eval.holdout_frac`) is never used by any job, script or analysis in the loop.
   Do not lower it, move it, or add a code path that builds rows from it.
2. No look-ahead. `pytest -m leakage` must pass before every push. Never skip, delete or loosen those tests.
3. If folds, baselines, targets or metrics change, bump `eval.version` in `configs/loop.yaml` in the
   same commit. Old scores then stop competing; do not hand-edit results.
4. Report honestly. A tier has "beaten the baseline" only if its champion's skill is positive on both
   targets or clearly positive on one with DM p < 0.05, and that holds on re-runs with other seeds.
   Null results are results; write them down.
5. Push only to branch `loop` of `chevvyy/forecast-engine` and `chevvyy/forecast-chat`. Never to `main`.
   No secrets in the repo (it is public). No new accounts, purchases, deployments or cluster changes.
6. Targets stay returns and realized volatility as quantiles (ADR-0001). No price-level targets.
   This is a research project: no trading, no investment advice.

## Each cycle
1. **Sync and test.** Check out `loop`, install deps, run `pytest -q` and `ruff check .`.
2. **Health.**
   - `loop/workers/*.json`: a worker whose `seen` is older than 6 h is down. Note it for the summary.
   - `results/*.json` with `status: error`: read `error`/`traceback`, fix the cause in code (with a test),
     and re-queue the job if it was a queued one. If every recent result fails the same way, fixing
     that is the whole cycle.
   - `jobs/running/*.yaml` whose worker has pushed newer results, or older than 3 h: move back to
     `queued/` (delete worker-invented ones, `origin: worker`).
3. **Leaderboard.** `python -m forecast.loop.leaderboard`, commit `experiments/LEADERBOARD.md`, and add
   one row per cycle to `experiments/LOG.md` (date, runs since last cycle, champion per tier with score,
   what was learned). Include the total number of configurations tried so far.
4. **Decide the next experiments** from the results, and write 3-8 job files into `jobs/queued/`:
   - re-run each new champion with 2 more seeds before trusting it (noise estimate);
   - distil the large champion into mid, and mid into nano;
   - prune-and-finetune each champion at 2 sparsity levels;
   - targeted variations of whatever the last results say matters (lookback, kernel, width, lr...).
   Workers fill idle time with random mutations on their own, so queue only deliberate experiments.
5. **Improve the code: the top unfinished backlog item**, with tests. Workers restart themselves when
   `src/` or `pyproject.toml` changes, so a broken push stops all training: run the full test suite and
   a CPU dry run (`data.source: synthetic`) of any changed job path before pushing.
6. **Housekeeping.** Remove `models/<id>/` folders that are no longer champion or runner-up of a tier.
   Keep `jobs/done/` and `results/` (they are the experiment record).
7. **Push** to `loop`. If the push is rejected, `git pull --rebase` and retry; never force-push.
8. **Tell Dev only when it matters:** a worker down > 6 h, the loop blocked on something only he can
   do, or a tier's champion beating the baseline for the first time (or by a new clear margin).

## Backlog (ordered; tick items off here as they land)
- [ ] First real-data triage: confirm jobs succeed on the real workers with Yahoo data; fix whatever
      broke (paths on Windows, VRAM accounting, data gaps). Until `results/` shows `ok` runs from a real
      GPU, this outranks everything else.
- [ ] **Wide universe (Dev asked for as many tickers as possible, 2026-11-25).** Two parts, in order:
      (a) lazy window loading: keep per-symbol channel arrays and index windows at batch time instead of
      materialising `X[n, C, L]`, which at 500+ symbols and lookback 256 needs ~15 GB of RAM; identical
      outputs to the current builder must be asserted in a test. (b) a committed symbol list
      `configs/data/universe_wide.txt` (S&P 500 + Nasdaq-100 + the ETFs already used; verify the list
      against a current source, do not write it from memory), downloaded politely (Yahoo starts returning
      429 after several hundred rapid requests: keep the per-symbol pause, back off on 429, reuse the
      parquet cache). Record in results that current-constituent lists carry survivorship bias. A new
      universe changes the folds' contents: bump `eval.version`.
- [ ] **Minute bars (Dev asked, for short-horizon forecasts).** Alpaca ingestor per docs/DATA_SOURCES.md:
      resumable, per-symbol-month parquet, `feed=sip`, end time at least 15 minutes old (free plan),
      under 200 requests/min, regular session only, switched on when `ALPACA_API_KEY` exists in a
      worker's `.env` (Dev has to create the keys; log under "Needs Dev" until a worker reports them).
      Separate config (`configs/loop_minute.yaml`: horizons 5/15/30/60 bars) and separate leaderboard.
      Until keys exist, the keyless stopgaps are Yahoo `1h` (~730 days) and `5m` (~60 days); 1-minute
      history there is only about the last 30 days, too short to validate anything.
      Intraday evaluation must also report results net of a per-trade cost assumption (docs/EVALUATION.md).
- [ ] Seed-noise estimate per tier; promotion must beat the champion by more than that noise.
- [ ] LightGBM quantile baseline on tabular features (the scaffold's "first real model"); deep models
      must be compared against it in every result. Bumps `eval.version`.
- [ ] Richer causal features: sector-ETF context per stock, cross-sectional ranks, calendar encodings.
- [ ] Serving: a small read-only FastAPI app in `src/forecast/serving/` that loads a tier champion from
      `models/` and returns quantiles + its validation metrics; then, on a `loop` branch of
      forecast-chat, a provider that calls it. Code only; deployment is Dev's or his cluster agent's.
- [ ] Phone tier: int8 quantisation and structured (channel) pruning, each kept only if skill holds.
- [ ] Result cards (`experiments/cards/`) for each champion; ADRs for design changes.
- [ ] Model variants behind the same interface: attention (PatchTST-style) block, ensembles of fold
      models, then TiRex/Kronos challengers on the 5070 Ti (Phase 4 of docs/ROADMAP.md).
