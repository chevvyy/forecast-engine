# ADR-0005: The `loop` git branch is the job queue and experiment store

**Status:** accepted (2026-11-25). Supersedes ADR-0004 for now.

**Context.** ADR-0004 chose a shared Optuna study in Postgres plus MLflow. Neither service exists yet,
the two GPUs sit on different machines (Windows desktop with the 5070 Ti, Proxmox guest with the 3060),
and the planner is a cloud session that cannot reach the home network. GitHub is the one place all three
can already reach.

**Decision.** Jobs are YAML files and results are JSON files on the `loop` branch. A worker claims a job
by moving it to `jobs/running/` and pushing; a rejected push means another worker won the race. Champions
and the leaderboard are derived from `results/`, never stored, so concurrent workers cannot conflict.
Workers generate their own experiments when the queue is empty.

**Consequences.** No infrastructure to run; everything is auditable in git history. Throughput is limited
to roughly one job per minute per worker, which is far above what training allows. Only champion weights
are committed; other checkpoints stay on the machine that trained them. If run volume outgrows git,
move to Optuna/MLflow as ADR-0004 describes; the job and result schemas map onto trials directly.
