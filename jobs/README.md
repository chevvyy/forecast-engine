# Job queue

One YAML file per job. `queued/` -> `running/` (claimed by a worker) -> `done/`, with the outcome in
`results/<id>.json`. Workers invent their own jobs when `queued/` is empty, so this folder is for
experiments someone specifically wants.

```yaml
id: 20261125-champ-seed2        # unique; also the artifact folder name
tier: nano | mid | large        # budget the run must fit (configs/loop.yaml -> tiers)
config: loop_minute             # optional track: configs/<name>.yaml instead of configs/loop.yaml (daily).
                                # Each track has its own eval.version, champions and leaderboard file.
kind: train | distill | prune   # label only; behaviour comes from the fields below
model: {d_model: 48, kernel: 15}   # overrides on top of the tier preset
train: {lr: 0.003}                 # overrides on top of configs/loop.yaml -> train
teacher: <run id>               # distil from this run's fold models (needs its artifacts)
init_from: <run id>             # start from this run's weights (prune / finetune); the job then has
                                # the parent's model shape and lookback, whatever `model:` says
sparsity: 0.5                   # with init_from: magnitude-prune this share, then finetune
seed: 1337
max_minutes: 20                 # wall-clock budget across all folds
min_vram_gb: 2                  # skip on machines with less free VRAM right now
```

Artifacts of non-champion runs stay on the machine that trained them (`artifacts/`, git-ignored).
A run that becomes its tier's champion has its fold checkpoints and ONNX file committed to `models/<id>/`.
