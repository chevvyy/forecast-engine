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
   - Same data everywhere: every `ok` result of the current `eval.version` should carry one `data_hash`
     and one `symbols` count. If they differ (a symbol failed to download on one machine or one day),
     say so in the log and do not compare those runs with each other.
   - `jobs/running/*.yaml` whose worker has pushed newer results, or older than 3 h: move back to
     `queued/` (delete worker-invented ones, `origin: worker`).
3. **Leaderboard.** `python -m forecast.loop.leaderboard`, commit `experiments/LEADERBOARD.md`, and add
   one row per cycle to `experiments/LOG.md` (date, runs since last cycle, champion per tier with score,
   what was learned). Include the total number of configurations tried so far.
   Once the minute track has results: `python -m forecast.loop.leaderboard --config loop_minute` too
   (`experiments/LEADERBOARD_minute.md`). Minute and daily scores are never compared with each other.
4. **Decide the next experiments** from the results, and write 3-8 job files into `jobs/queued/`:
   - re-run each new champion with 2 more seeds before trusting it (noise estimate);
   - distil the large champion into mid, and mid into nano;
   - prune-and-finetune each champion at 2 sparsity levels;
   - targeted variations of whatever the last results say matters (lookback, kernel, width, lr...).
   Workers fill idle time with random mutations on their own, so queue only deliberate experiments.
5. **Improve the code: the top unfinished backlog item**, with tests. Workers restart themselves when
   `src/` or `pyproject.toml` changes, so a broken push stops all training: run the full test suite and
   a CPU dry run (`data.source: synthetic`) of any changed job path before pushing.
6. **Housekeeping.** Remove `models/<id>/` folders that are no longer champion or runner-up of a tier
   and are not the `teacher` / `init_from` of a job still in `jobs/queued/`.
   Champion and runner-up are per track and version (daily, minute, and the closed version 1). Also keep
   the first run of a configuration the log calls confirmed across seeds, while it is the confirmed
   one for its tier (since 2026-11-27 01:10: nano `20261126-x3-nano-champ-lr0003-s11`, mid
   `20261127-a3-mid-k9-batch2048-lr0003-s11`, large `20261126-190159-pc-5070ti-01c4`; minute-2: nano
   `20261127-a6-minute2-nano-s11`, mid `20261127-a8-minute2-mid-s11`).
   Keep `jobs/done/` and `results/` (they are the experiment record).
7. **Push** to `loop`. If the push is rejected, `git pull --rebase` and retry; never force-push.
8. **Tell Dev only when it matters:** a worker down > 6 h, the loop blocked on something only he can
   do, or a tier's champion beating the baseline for the first time (or by a new clear margin).

## Prune results before 2026-11-26 09:20 (do not use)
A prune job (`init_from`) built its windows at the tier preset's lookback, not its parent's, and recorded
the preset shape. Where the parent's lookback is longer than the preset's, the parent was finetuned and
scored on windows shorter than it was trained on: version 2 nano runs x6, x7 and a353 (parent lookback
256, windows 64), and any version-1 prune of a champion with a longer lookback than its preset. Large
prunes (preset 256, parents 128) were computed correctly; only their recorded `model` is the preset.
The result files are left as they are (results are not hand-edited); none of them is a champion.
Fixed in `run_job` with `test_prune_job_uses_its_parents_shape_and_lookback`; x6 / x7 re-queued as y6 / y7.

## Data source note (2026-11-25)
Yahoo answered HTTP 429 to the first request from pc-5070ti: it refuses scripted clients. Do not add
browser impersonation, spoofed TLS fingerprints, proxies or any other way around that refusal. Daily
bars now come from Alpaca when the worker has keys (`data.source: auto`), from 2016-01-04. Results from a
worker without keys (Yahoo) and one with keys (Alpaca) differ slightly in data; if both kinds ever
report, record the source in results and keep them on separate leaderboards. The wide-universe list
still applies; with Alpaca, 539 symbols is about 4 minutes of requests at the free plan's rate.

## Data pin and evaluation versions (2026-11-26)
`data.end` in `configs/loop.yaml` is the last bar any job may load. With it, cached daily files are never
refetched and every job of an `eval.version` sees the same bars and folds. Move `data.end` forward only
together with an `eval.version` bump (new bars move the holdout and every fold). Version history:
1 = 35 symbols, closed 2026-11-26, final board in `experiments/LEADERBOARD_v1.md`;
2 = 539-symbol list (`configs/data/universe_wide.txt`), data to 2026-11-25.
`experiments/LEADERBOARD.md` shows the current version only.
Minute track: `minute-1` = unconditional and HAR baselines, no clock (2 runs, closed 2026-11-26, board in
`experiments/LEADERBOARD_minute_v1.md`); `minute-2` = time-of-day inputs and the HAR time-of-day baseline.

## Backlog (ordered; tick items off here as they land)
- [x] First real-data triage (done 2026-11-25 21:15): `ok` runs from pc-5070ti on Alpaca daily bars since
      2026-11-26 00:22 UTC. Two breakages found and fixed: missing `tzdata` on Windows (1064 failed
      runs), and VRAM accounting: the worker counted its own cached GPU memory as busy, so after its
      first 6 GB job it ran only nano jobs (`common.usable_vram_gb`, cache emptied after every job).
      Paths and data gaps: nothing seen (all 35 symbols usable, identical folds in every run).
      Next cycle must confirm the VRAM fix on the real machine: `vram_free_gb` in the heartbeat back
      near 14, and mid/large results arriving again. If not, something else is using that GPU.
- [x] **Wide universe (Dev asked for as many tickers as possible, 2026-11-25).** Three parts, in order:
      (a) [x] lazy window loading (done 2026-11-25: `features.build.Windows`; `tests/test_windows.py`
      asserts equality with the old eager builder; 520 synthetic symbols x 5400 bars at lookback 256 build
      in ~0.5 GB instead of 15 GB). (b) [x] committed symbol list and polite download (done
      2026-11-25: `configs/data/universe_wide.txt`, 539 symbols = S&P 500 + Nasdaq-100 + the 15 ETFs,
      sources and how far each was cross-checked are in the file header; `data.symbols_file` is read by
      `sources.resolve_symbols`; Yahoo 404 skips the symbol at once, 429 backs off 15 s -> 120 s and
      honours Retry-After, and after repeated 429s the rest of the call uses the parquet cache; every
      result now carries `universe.requested` and the survivorship note; `tests/test_data_sources.py`).
      (c) [x] switched (done 2026-11-26 01:10 cycle): `data.symbols_file` on, `eval.version` 2, `data.end`
      pinned to 2026-11-25, tier budgets raised to 15/30/60 minutes (about 18x more rows per epoch; a
      job that hits its budget stops early and reports fewer `epochs`). Alpaca requests use its spelling
      (`BRK.B`), and a symbol it does not know is marked and not asked for again. Only ever run on
      synthetic data so far. **Next cycle must check on the real machine:** `symbols` in the first
      results (expect a little under 539), one `data_hash` throughout, `minutes` and `epochs` against
      the budgets, and that no score is NaN. If every job fails, revert `configs/loop.yaml` to the
      35-symbol universe (and bump `eval.version` again) rather than leaving the GPU idle.
      Still open: the list was not re-verified against a second full source (281 of the 503 S&P rows
      rest on one; Wikipedia could not be fetched from the planner on 2026-11-26 either), and what
      Alpaca really answers for an unknown symbol is assumed (400/404/422 or an empty page), not seen.
- [x] **Worker-invented jobs: seeds, no repeats, VRAM guard, fewer commits, no NaN champions** (done
      2026-11-26 05:15; found 2026-11-25 when 142 distil runs were 11 distinct jobs). `worker.propose` now
      gives every proposal a seed and redraws one that (a) already has a result of this `eval.version`
      (`worker.signature`: tier, kind, model and train with presets filled in, teacher/parent, sparsity,
      seed; failed results count), (b) is a distil / prune configuration already run `MAX_REPEATS` = 3
      times, or (c) would not fit the tier's `max_train_vram_gb` by `patchnet.train_vram_gb`
      (0.8 GB + batch x tokens x d_model x blocks x (8 + 8 ffn) bytes, fitted to pc-5070ti's peaks: of 49
      past mid/large runs it would have refused the 2 that did not fit and none that did; it refuses 246
      of the 1536 large grid points and 55 of 2880 mid). After 20 redraws it re-runs the champion's shape
      with a new seed. A self-invented job in a tier with a budget of 20 minutes or less (nano) makes one
      commit, not a claim plus a result; queued jobs and long jobs are still claimed first, so a running
      mid/large job stays visible. A non-finite loss is a failed run (`run_job`), and `champions` ignores
      a non-finite score. `search.p_prune` is back at 0.15; `p_distill` stays 0.1 on purpose (see the
      comment in `configs/loop.yaml`). Correction to the earlier note here: at batch 512 a 255-token,
      8-block, ffn 4 model needs about 16 GB at d_model 384 (measured 16.6), not 10.
      **Next cycle must check on the real machine** that the worker restarted on this code (results'
      `git` newer than this commit), that self-invented distil / prune results carry `job.seed`, and
      that nano results arrive without a preceding "claims" commit.
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
      (b2a) [x] minute track (done 2026-11-26 09:20): a queued job with `config: loop_minute` runs under
      `configs/loop_minute.yaml` (`run_job.job_cfg`; the name must be a plain file stem in `configs/`).
      10 hand-picked liquid symbols from 2026-06-01, pinned to 2026-11-25; horizons 5/15/30/60 bars;
      `max_lookback` 120, so each symbol-day has 390 - 120 - 60 + 1 = 211 forecast times (11:29 to
      14:59 New York). Re-derived for minute bars: `embargo_bars` 211 (one session of forecast times
      between the last training label and the first validation row, because minutes of one day share
      that day's volatility level) and the new `eval.dm_lag` 211 (label overlap needs 60; a session also
      covers within-day correlation of losses). Its `eval.version` is the string `minute-1`; champions,
      worker proposals and leaderboards are all per version, so the tracks cannot compete, and
      `tests/test_loop.py` keeps the versions distinct. Workers never invent minute jobs (until part (c),
      2026-11-26 21:15). Own board:
      `leaderboard --config loop_minute` -> `experiments/LEADERBOARD_minute.md`.
      Only ever run on synthetic sessions. Job y8 is the first on real Alpaca minute bars.
      **Next cycle must check on the real machine:** y8's `status` (the ingestor has never talked to the
      real API; an error there is the expected kind of finding), `symbols` = 10, rows per fold against
      10 x sessions x 211, one `data_hash` across minute runs, and whether the baselines (unconditional,
      HAR on 20-bar realised volatility) are sensible at one minute before reading any skill number.
      **Checked 2026-11-26 13:10 on the real machine:** y8 ran cleanly on its first contact with the real
      API (2 minutes). 10 symbols; 162,930 / 180,400 / 198,340 training rows and 19,410 validation rows
      per fold (9.2 sessions x 10 x 211; a whole number of sessions is not expected, a missing minute
      costs rows); HAR beats the unconditional volatility baseline by 9.5% (0.0919 against 0.1015), so
      the baselines behave; 80% coverage 0.78-0.80; training ran 9-10 epochs per fold. Score +0.0333
      (vol +0.062, return +0.005, rank IC 0.000), one seed: read it only after item (2) below, because
      forecasts sit between 11:29 and 14:59 and HAR does not know the time of day.
      (b2b) [ ] still open, in this order: (1) [x] the pin inside the fetch (done 2026-11-26 13:10):
      `alpaca.load(..., end=)` asks for no bar at or after the pin, skips ranges with no session minute,
      and marks the open month as asked up to the pin (`YYYY-MM.through`), so a pinned job makes no bar
      request after the first; adjustment factors are normalised to the last day before the pin and
      cached per symbol and pin under `alpaca/factors/` (`pinned_factors`), so a split or dividend after
      the pin cannot change the bars, on a machine that already has them or on one that fetches them
      later (`test_pinned_minute_load_fetches_nothing_past_the_pin_and_freezes_the_adjustment`). Same
      bars as before as long as nothing went ex-dividend on 2026-11-26 or since: **next cycle must check
      that z8 reports `data_hash` b1f8ab268f40**; if it differs, y8 and z8 are not comparable, say so in
      the log and re-run y8's seed. **Checked 2026-11-26 17:15:** z8 reports b1f8ab268f40, the same bars
      as y8; it scored +0.0329 against y8's +0.0333;
      (2) [x] time-of-day inputs and a volatility baseline that knows the shape of the session (done
      2026-11-26 17:15): intraday rows carry `tod` (share of the 09:30-16:00 New York session elapsed
      when the row becomes usable) and `tod_u` = (2 tod - 1)^2 as two more side inputs
      (`features.build.TAB_TOD`, from the clock alone: leakage test), and volatility is scored against
      the best of unconditional, HAR and `baselines.HarTod` (HAR plus one level per 15-minute bin, per
      horizon: a multiplicative time-of-day factor on forward / trailing volatility). The result names
      the winner in `vol_baseline` and carries `pinball.vol_har_tod`. Baselines changed, so the track is
      now `minute-2`; y8 / z8 stay on the closed `minute-1` board and their checkpoints (2 side inputs)
      are refused as teacher or parent of a minute-2 job. The daily track is untouched (dry run
      bit-identical, no version bump). On synthetic sessions, which have no time-of-day pattern, HarTod is
      about 2% worse than HAR, as it should be. **Next cycle must check on the real machine (a6, a7,
      a8):** how much `vol_har_tod` improves on `vol_har` (if not at all between 11:29 and 14:59, say
      so), which baseline `vol_baseline` names, and only then the scores; expect them below minute-1's
      +0.033 if part of that was the clock. Known gap: on a half-day the time of day does not know the
      early close (item 3). **Checked 2026-11-26 21:15 (a6, a7, a8):** `vol_har_tod` 0.0899 against
      `vol_har` 0.0919 (2.2% lower loss, so the session's shape matters even between 11:29 and 14:59) and
      every run names `har_tod` as its volatility baseline. Against it nano scores +0.0256 / +0.0266
      (vol +0.048, minute-1 had +0.062 against plain HAR: about a quarter of that was the clock), mid
      +0.0298 on one seed. Same data hash b1f8ab268f40;
      (3) [x] exchange calendar for half-days (done 2026-11-27 01:10): `forecast.data.calendar` holds the
      early closes 2016-2028 as a table (2026-2028 from the NYSE Group calendar announced 2025-12-23;
      2016-2025 from memory, unchecked, and loaded by no minute track). On such a day
      `alpaca.regular_session` keeps 09:30-13:00 only (210 bars), when bars are fetched and again on
      load, so a month stored earlier loses its late-session bars too, and `time_of_day` is the share of
      that day's own session. Published in advance, so not look-ahead (leakage test). The minute-2 window
      (2026-06-01 to 2026-11-25) contains NO early close (2026-07-02 was a full day; the first one is
      2026-11-27, after the pin), so no bar and no input changes: no version bump, a test
      (`tests/test_calendar.py`) fails if a minute config's window ever contains one under `minute-2`, and
      the CPU dry run on synthetic sessions is bit-identical before and after. Whoever moves the minute
      `data.end` past 2026-11-27 bumps the version anyway (pin rule). **Next cycle must check on the real
      machine:** minute results with `finished` after 2026-11-27 08:40 UTC (c7 is one) still report
      `data_hash` b1f8ab268f40 and the same rows per fold (19,410 validation). If the hash differs, the
      stored bars held something the calendar now drops, which I did not expect: bump to `minute-3`
      and say so. Not covered: unscheduled closures, and dates after 2028 (treated as full days; the
      test keeps the pins inside the table);
      (4) out-of-core windows before the symbol list grows; (5) results net of a per-trade
      cost (docs/EVALUATION.md section 4). That last one needs a backtest of a simple rule, which the
      loop's standing rules exclude (no trading code), so it waits for Dev to say what he wants there.
      Can also be tried on Yahoo `1h`/`5m` (keyless stopgaps: ~730 and ~60 days; note Yahoo `5m` needs
      `range=60d`, the fetcher asks for 730d; at `1h` a session is 7 bars, so only lookback + horizon
      <= 7 fits).
      (c) [x] switch workers that have keys onto minute jobs (done 2026-11-26 21:15, ahead of (b2b)
      items 3-5: its own conditions, y8 clean and (b2b) 1 and 2 in, were met, and item 3 changes at most
      one session of the pinned window). `search.tracks: {loop_minute: 0.3}` in `configs/loop.yaml` gives
      30% of a worker's self-invented jobs, by count, to the minute track; `worker.pick_track` takes the
      share only if the track's config loads, has its own `eval.version` and `search` grid, and the
      machine has Alpaca keys, else the job is a daily one. The job is drawn from the minute track's own
      champions and grid (`configs/loop_minute.yaml` -> `search`, lookbacks 30-120), carries
      `config: loop_minute` and lands on the minute board only. Evaluation untouched, no version bump.
      Only ever run on synthetic sessions. **Next cycle must check on the real machine:** results with
      `job.origin: worker` and `config: loop_minute` exist, all `eval_version` minute-2 with data hash
      b1f8ab268f40 and 10 symbols; the first one is the large preset (no large minute run yet); roughly
      3 in 10 self-invented jobs are minute jobs; none failed; daily results keep arriving. If minute
      jobs fail or crowd out daily ones, set `search.tracks: {}` (a config edit, no restart needed).
      When (b2b) item 3 bumps the minute version, the worker's minute runs restart from the presets.
- [ ] Seed-noise estimate per tier; promotion must beat the champion by more than that noise.
      First number (2026-11-25 21:15): the nano champion's configuration over 15 seeds scores mean
      +0.022, sd 0.005, range +0.014..+0.031. The leaderboard champion is the best of those seeds, so
      its own score overstates the configuration by about 0.009. Not yet in code: the leaderboard
      should rank configurations by their seed mean, and `propose` should re-run a would-be champion.
      On version 2 the noise looks larger: two seeds of one nano shape scored +0.009 and -0.007.
      Version 2 numbers (2026-11-26 13:10): sd across seeds of one configuration is 0.003-0.004 in nano
      and mid where a shape has 3-4 seeds (nano batch 2048: +0.044, +0.038, +0.037; mid a7c6 shape:
      +0.017, +0.012, +0.018, +0.019). The
      board still ranks single runs: the mid and large "champions" (066b, 01c4) are one seed each.
      Same function, found 2026-11-26: a mutation starts from the champion's model shape but not from
      its training settings (`lr`, `weight_decay`, `batch_size` fall back to the defaults unless they
      are the mutated keys), so the search cannot hill-climb on those. Inherit `job.train` of the champion.
      This now costs something (2026-11-26 09:20): the nano champion is the old shape at batch 2048, and
      every mutation of it goes back to batch 512.
      Early stopping is too coarse for the wide universe (same date): with about 1800 steps per epoch,
      every fold of 13 of the 29 version-2 training runs kept its FIRST epoch (`epochs` 6 = 1 + patience),
      also at lr 0.0003 and at batch 2048, both of which scored better. The best point is probably inside
      the first epoch. Evaluate the early-stopping slice every N steps (or a fraction of an epoch) and
      record the best step in `folds`. This changes training, not the evaluation: no version bump.
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
