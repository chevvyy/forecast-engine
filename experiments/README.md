# experiments

- `LOG.md` — append-only, one line per experiment batch. The total count of configurations tried feeds the
  Deflated Sharpe Ratio, so log failures too.
- `cards/` — one markdown result card per reportable run.

## Result card template (copy to `cards/YYYY-MM-DD-<model>-<target>.md`)

```markdown
# <model> — <target> — <horizon(s)>
- MLflow run: <id>    Git: <sha>    Data hash: <hash>    Config hash: <hash>
- GPU: <name>         Wall time: <min>    Max VRAM: <GB>
- Folds: <n> walk-forward, purge=<..>, embargo=<..>

| Metric | Model | Best baseline | Δ | DM p-value |
|---|---|---|---|---|
| CRPS / QLIKE | | | | |
| RankIC | | | | |
| Hit rate (binomial p) | | | | |
| Coverage 80% | | | | |

Per-fold stability: <table or chart>
Backtest (net of 2 bps/side): Sharpe <..>, MDD <..>, turnover <..>, DSR <..>
Conclusion: <one paragraph, including "no edge" if so>
```
