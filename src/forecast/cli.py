"""CLI: `python -m forecast <command>` or the fe-* entry points."""
from __future__ import annotations

import argparse
import json
import sys


def _cfg(a):
    from forecast.training.experiment import load_run_config
    return load_run_config(a.config)


def cmd_ingest(a):
    from forecast.data.ingest import ingest
    from forecast.training.experiment import universe_symbols
    from forecast.utils.timeutil import utc
    cfg = _cfg(a)
    u = cfg["universe_cfg"]
    syms = a.symbols.split(",") if a.symbols else universe_symbols(u)
    print(ingest(syms, utc(a.start or str(u["start"])), utc(a.end) if a.end else None,
                 a.timeframe or cfg["data"]["timeframe"], feed=u.get("feed", "sip")))


def cmd_dq(a):
    import polars as pl

    from forecast.data.align import session_align
    from forecast.data.quality import dq_report
    from forecast.data.store import read_bars
    cfg = _cfg(a)
    tf = cfg["data"]["timeframe"]
    adj = session_align(read_bars("adjusted", tf), tf)
    try:
        raw = read_bars("raw", tf)
    except FileNotFoundError:
        raw = None
    with pl.Config(tbl_rows=200, tbl_cols=20):
        print(dq_report(adj, raw))


def cmd_baselines(a):
    from forecast.evaluation.report import write_card
    from forecast.models.baselines import baselines_for
    from forecast.training import experiment as E
    cfg = _cfg(a)
    fs, h = E.prepare(cfg)
    fl = E.folds(fs, cfg)
    for name, task in E.make_tasks(fs, cfg).items():
        res = {}
        for b in baselines_for(task):
            o = E.run_oof(lambda b=b: type(b)(), fs, task, fl, fs.feature_cols)
            res[o.model] = (o, E.score(o))
        ranked = sorted(res, key=lambda k: res[k][1]["qlike" if task.kind == "rv" else "crps"])
        best, second = ranked[0], ranked[1]
        dm = E.compare(res[best][0], res[second][0])
        write_card(task, best, res[best][1], second, res[second][1], dm,
                   {"data_hash": h, "n_folds": len(fl), "embargo_min": cfg["eval"]["embargo_min"],
                    "n_trials_total": len(res)}, "Baseline reference card.", tag="-baseline")
        print(name, {k: {kk: v for kk, v in s.items() if kk != "per_fold"} for k, (_, s) in res.items()})


def cmd_autotune(a):
    from forecast.training.autotune import run
    run(_cfg(a), promote=True, rounds=a.rounds)


def cmd_worker(a):
    from forecast.training.autotune import run
    run(_cfg(a), promote=False, rounds=a.rounds)


def cmd_holdout(a):
    from forecast.training.holdout import run
    print(json.dumps(run(_cfg(a), a.task, a.force), indent=2, default=float))


def cmd_score(a):
    from forecast.serving.batch import score_latest
    out = score_latest(_cfg(a))
    print(f"scored {len(out['symbols'])} symbols")


def cmd_serve(a):
    import uvicorn
    uvicorn.run("forecast.serving.api:app", host=a.host, port=a.port)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="forecast")
    p.add_argument("--config", default="configs/run/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest")
    for opt in ("--symbols", "--start", "--end", "--timeframe"):
        s.add_argument(opt)
    s.set_defaults(f=cmd_ingest)
    sub.add_parser("dq").set_defaults(f=cmd_dq)
    sub.add_parser("baselines").set_defaults(f=cmd_baselines)
    for n, f in (("autotune", cmd_autotune), ("worker", cmd_worker)):
        s = sub.add_parser(n)
        s.add_argument("--rounds", type=int)
        s.set_defaults(f=f)
    s = sub.add_parser("holdout")
    s.add_argument("--task", required=True)
    s.add_argument("--force", metavar="REASON")
    s.set_defaults(f=cmd_holdout)
    sub.add_parser("score").set_defaults(f=cmd_score)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8088)
    s.set_defaults(f=cmd_serve)
    a = p.parse_args(argv)
    a.f(a)


def _entry(cmd: str):
    """fe-<cmd> [--config X] [args...] -> forecast --config X <cmd> [args...]"""
    def run():
        argv, pre = list(sys.argv[1:]), []
        if "--config" in argv:
            k = argv.index("--config")
            pre = argv[k:k + 2]
            del argv[k:k + 2]
        main([*pre, cmd, *argv])
    return run


fe_ingest, fe_dq, fe_baselines = _entry("ingest"), _entry("dq"), _entry("baselines")
fe_autotune, fe_worker, fe_holdout = _entry("autotune"), _entry("worker"), _entry("holdout")
fe_score, fe_serve = _entry("score"), _entry("serve")


if __name__ == "__main__":
    main()
