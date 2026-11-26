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
   Install `tzdata` too (`pip install tzdata`), or the Windows zone-database test is skipped.
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

## Data source note (2026-11-25)
Yahoo answered HTTP 429 to the first request from pc-5070ti: it refuses scripted clients. Do not add
browser impersonation, spoofed TLS fingerprints, proxies or any other way around that refusal. Daily
bars now come from Alpaca when the worker has keys (`data.source: auto`), from 2016-01-04. Results from a
worker without keys (Yahoo) and one with keys (Alpaca) differ slightly in data; if both kinds ever
report, record the source in results and keep them on separate leaderboards. The wide-universe list
still applies; with Alpaca, 539 symbols is about 4 minutes of requests at the free plan's rate.

## Backlog (ordered; tick items off here as they land)
- [x] First real-data triage (done 2026-11-25 21:15): `ok` runs from pc-5070ti on Alpaca daily bars since
      2026-11-26 00:22 UTC. Two breakages found and fixed: missing `tzdata` on Windows (1064 failed
      runs), and VRAM accounting: the worker counted its own cached GPU memory as busy, so after its
      first 6 GB job it ran only nano jobs (`common.usable_vram_gb`, cache emptied after every job).
      Paths and data gaps: nothing seen (all 35 symbols usable, identical folds in every run).
      Next cycle must confirm the VRAM fix on the real machine: `vram_free_gb` in the heartbeat back
      near 14, and mid/large results arriving again. If not, something else is using that GPU.
- [ ] **Wide universe (Dev asked for as many tickers as possible, 2026-11-25).** Three parts, in order:
      (a) [x] lazy window loading (done 2026-11-25: `features.build.Windows`; `tests/test_windows.py`
      asserts equality with the old eager builder; 520 synthetic symbols x 5400 bars at lookback 256 build
      in ~0.5 GB instead of 15 GB). (b) [x] committed symbol list and polite download (done
      2026-11-25: `configs/data/universe_wide.txt`, 539 symbols = S&P 500 + Nasdaq-100 + the 15 ETFs,
      sources and how far each was cross-checked are in the file header; `data.symbols_file` is read by
      `sources.resolve_symbols`; Yahoo 404 skips the symbol at once, 429 backs off 15 s -> 120 s and
      honours Retry-After, and after repeated 429s the rest of the call uses the parquet cache; every
      result now carries `universe.requested` and the survivorship note; `tests/test_data_sources.py`).
      (c) [ ] switch the loop to it: uncomment `data.symbols_file` in `configs/loop.yaml` and bump
      `eval.version` in the same commit. Do this only after `results/` shows `ok` real-data runs on the
      35-symbol universe (item 1), so the first triage is not confounded by a 15x larger download and
      dataset. When switching, check the tier time budgets: the first download is roughly 539 requests
      (several minutes) and an epoch is about 15x longer, so `max_minutes` may need raising, and the
      list should be re-verified against a second full source (281 of the 503 S&P rows rest on one).
- [ ] **Minute bars (Dev asked, for short-horizon forecasts).** Three parts, in order:
      (a) [x] Alpaca ingestor (done 2026-11-25: `forecast.data.alpaca`, `data.source: alpaca`;
      resumable per-symbol-month parquet, `feed=sip`, nothing newer than 16 minutes, ~170 requests/min,
      regular session only; `tests/test_alpaca.py` runs it against a fake server). It stores RAW bars and
      adjusts on load from daily adjusted/raw closes instead of requesting `adjustment=all`, because a
      cache of provider-adjusted months goes stale at the next split or dividend. It has never talked to
      the real API (no keys yet): the first worker that reports `alpaca_keys: true` in
      `loop/workers/<name>.json` must be given one small ingest job before anything depends on it.
      Known gaps, written in the module docstring: no exchange calendar (half-day afternoons keep a few
      after-hours bars), missing minutes are not filled, `load` holds everything in memory.
      (b) feature/label path, then config and scoring:
      (b1) [x] intraday feature/label path (done 2026-11-25: `features.build.build(..., bar_seconds=)`,
      chosen by `data.interval` in the job runner; `tests/test_intraday.py`, 4 leakage tests). A bar's
      `t` is its START, so a row is usable only from `Dataset.known` = `t` + one bar. A row exists only
      if its window bars and label bars sit on one unbroken bar grid, so nothing crosses the overnight
      gap, a weekend or a missing bar; at a session open the return is open-to-close and the gap channel
      is 0. Trailing statistics (rv20 scale, volume z-score, HAR regressors) still run across sessions on
      intraday returns. Costs of the rule: no forecast in the first `max_lookback` - 1 bars of a session
      or its last `max(horizons)` bars, and a symbol with scattered missing minutes loses many rows.
      Only ever run on synthetic sessions; no real intraday bars have gone through it yet.
      (b2) [ ] `configs/loop_minute.yaml` (horizons 5/15/30/60 bars at 1 minute; `max_lookback` plus the
      longest horizon must fit well inside a 390-bar session) selected per job or per worker, with its
      own leaderboard so minute and daily scores never compete, and intraday evaluation that also reports
      results net of a per-trade cost assumption (docs/EVALUATION.md). The DM lag and `embargo_bars`
      need re-deriving for minute bars. Start with a short symbol list; hundreds of symbols times years
      of minutes needs an out-of-core window path first. Can be tried on Yahoo `1h`/`5m` (keyless
      stopgaps: ~730 and ~60 days; note Yahoo `5m` needs `range=60d`, the fetcher asks for 730d; at `1h`
      a session is 7 bars, so only lookback + horizon <= 7 fits). A time-of-day input belongs with the
      "richer causal features" item.
      (c) [ ] switch workers that have keys onto minute jobs. Needs Dev: create Alpaca paper keys and put
      `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in that worker's `.env` (log under "Needs Dev" until a
      heartbeat shows `alpaca_keys: true`).
- [ ] **Worker-invented distil and prune jobs carry no seed and are never checked against earlier runs**
      (found 2026-11-25 21:15): 142 distil runs were 11 distinct jobs and 84 prune runs were 23, each
      repeat giving the identical score. Give them a random seed and skip a proposal whose
      (tier, kind, model, train, seed, teacher/parent, sparsity) already has a result. Same change:
      nano jobs take ~15 s and each makes two commits (about 7 commits a minute, 3000+ in four hours),
      so let the worker commit claim and result together for self-invented jobs, or space them out.
- [ ] Seed-noise estimate per tier; promotion must beat the champion by more than that noise.
      First number (2026-11-25 21:15): the nano champion's configuration over 15 seeds scores mean
      +0.022, sd 0.005, range +0.014..+0.031. The leaderboard champion is the best of those seeds, so
      its own score overstates the configuration by about 0.009. Not yet in code: the leaderboard
      should rank configurations by their seed mean, and `propose` should re-run a would-be champion.
- [ ] LightGBM quantile baseline on tabular features (the scaffold's "first real model"); deep models
      must be compared against it in every result. Bumps `eval.version`.
      Same version bump: a fairer return baseline. Returns are scaled by rv20 and the return baseline
      is unconditional quantiles, so a model can earn "return skill" purely by forecasting volatility
      better than rv20 (wider or narrower quantiles), with no directional skill. Add a baseline that
      rescales the unconditional return quantiles by the HAR volatility forecast, and report return
      skill against the better of the two. Until then read return skill together with `rank_ic_h1`.
- [ ] Richer causal features: sector-ETF context per stock, cross-sectional ranks, calendar encodings.
- [ ] Serving: a small read-only FastAPI app in `src/forecast/serving/` that loads a tier champion from
      `models/` and returns quantiles + its validation metrics; then, on a `loop` branch of
      forecast-chat, a provider that calls it. Code only; deployment is Dev's or his cluster agent's.
- [ ] Phone tier: int8 quantisation and structured (channel) pruning, each kept only if skill holds.
- [ ] Result cards (`experiments/cards/`) for each champion; ADRs for design changes.
- [ ] Model variants behind the same interface: attention (PatchTST-style) block, ensembles of fold
      models, then TiRex/Kronos challengers on the 5070 Ti (Phase 4 of docs/ROADMAP.md).
