# Experiment log

| Date | Model | Target/horizon | # configs tried | Best val metric | vs baseline | Card |
|---|---|---|---|---|---|---|

## Planner cycles

One row per planner run (times America/Denver). Scores are skill versus the best baseline on purged
walk-forward validation; the final holdout is never used.

| Date | Runs since last cycle | Champion and score per tier | Configs tried (total) | What was learned / done | Needs Dev |
|---|---|---|---|---|---|
| 2026-11-25 01:30 | 0 (0 ok, 0 failed) | nano: none; mid: none; large: none | 0 | First cycle, about 1 h after the `loop` branch was created. No worker heartbeat in `loop/workers/` yet, so there are no results and nothing to score. Backlog item 1 (first real-data triage) needs real worker results and stays open; reading taken: do the next item that needs no GPU. Landed wide-universe part (a), lazy window loading: 520 synthetic symbols x 5400 bars at lookback 256 now build in ~0.5 GB RAM instead of a 15.2 GB array, values identical to the old builder (test), CPU dry run of train/distil/prune on synthetic gives the same scores before and after. Evaluation unchanged, so `eval.version` stays 1. Queued 6 jobs: each tier preset with seeds 11 and 12 (first real-data check plus seed-noise estimate). No message sent to Dev. | Start a worker: run the bootstrap command from the README on pc-5070ti (and server-3060 if wanted); nothing can train until one appears. Later, for minute bars: create Alpaca keys and put `ALPACA_API_KEY` in the worker's `.env`. |
