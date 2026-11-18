# The self-improving loop (`fe-autotune`)

## What "improving" means here
For each task (e.g. `rv_30` = realized vol over the next 30 min, `ret_30` = 30-min log return), the loop
searches **LightGBM hyperparameters and which feature groups to use**. A trial's score is its walk-forward
validation loss divided by the best baseline's loss, fold by fold (QLIKE for vol, CRPS for returns).

Every round:
1. Run `trials_per_round` Optuna trials (TPE sampler, median pruner kills bad trials after their first folds).
2. Take the round's best trial as a **challenger** and test it against the **champion** (initially the best
   baseline) on the per-timestamp loss series with a HAC Diebold–Mariano test.
3. Promote only if loss drops by `min_rel_improvement` **and** p < `promote_alpha` (default 0.01).
4. A promoted champion is refit on all research data, saved, gets a result card, a `LOG.md` line, and two
   perturbed neighbours are queued so the next round explores around it.

## Why it can't "maximise prediction rate" without limit
Optimising a metric over many configs on the same data fits noise. The guards:
- **Purged walk-forward**: no training label interval overlaps validation; extra 450-min gap.
- **Honest baselines**: return models must beat `zero_return_volscaled` (median 0, HAR-scaled spread).
  Beating plain zero-return only by forecasting the spread is a vol model, not a return model.
- **Promotion needs significance**, not just a lower number.
- **Trial count is logged** (`LOG.md`, cards) for the Deflated Sharpe Ratio.
- **Final holdout** (last 6 months) is never read by the loop. `fe-holdout` reads it once per champion;
  a second look needs `--force REASON` and is recorded in `experiments/holdout_lock.json`.
- **New data → new study**: champions are tied to a data hash and must re-earn the title.

## Realistic expectations
Volatility is forecastable; expect clear, significant wins over HAR-RV. Short-horizon returns are close to
unpredictable for liquid US equities: a tiny IC (~0.01–0.03) that survives costs would already be a good
result, and "no edge" is a likely and valid outcome. Directional hit rate is reported with a binomial test.

## Scaling out
Point `OPTUNA_STORAGE` at the Postgres on k3s and run `fe-worker` on the 3060 box; it adds trials to the
same studies. Run exactly one `fe-autotune` (it does promotions).
