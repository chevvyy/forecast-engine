from datetime import UTC, datetime

import polars as pl

from forecast.data.align import session_minutes
from forecast.data.ingest import ingest
from forecast.data.store import BAR_SCHEMA, read_bars
from forecast.data.synthetic import make_bars


def test_half_day_has_210_minutes():
    g = session_minutes(datetime(2024, 11, 27, tzinfo=UTC), datetime(2024, 12, 3, tzinfo=UTC))
    per = g.group_by("session").len().sort("session")
    lens = dict(zip(per["session"].cast(pl.Utf8), per["len"], strict=True))
    assert lens["2024-11-29"] == 210 and lens["2024-11-27"] == 390


def test_timestamps_utc(aligned):
    assert str(aligned.schema["ts"].time_zone) == "UTC"
    # first regular-session bar open = 09:30 ET = 14:30 UTC in January
    first = aligned.filter(pl.col("symbol") == "AAA")["ts"].min()
    assert (first.hour, first.minute) == (14, 30)


def _fake_fetch(calls):
    def fetch(sym, tf, a, b, adj, feed):
        calls.append((sym, a, adj))
        df = make_bars([sym], a, b, seed=a.month, drop_frac=0)
        return df.select([pl.col(c).cast(t) for c, t in BAR_SCHEMA.items()])
    return fetch


def test_ingest_idempotent_and_resumable(tmp_path):
    calls = []
    s, e = datetime(2023, 1, 3, tzinfo=UTC), datetime(2023, 3, 1, tzinfo=UTC)
    ingest(["AAA"], s, e, "1Min", layers=("adjusted",), root=tmp_path, fetch=_fake_fetch(calls))
    first = read_bars("adjusted", "1Min", root=tmp_path)
    n1 = len(calls)
    ingest(["AAA"], s, e, "1Min", layers=("adjusted",), root=tmp_path, fetch=_fake_fetch(calls))
    assert len(calls) == n1, "completed chunks must not be re-pulled"
    assert read_bars("adjusted", "1Min", root=tmp_path).equals(first)
    # simulate a crash: drop the last manifest line -> only that chunk is re-pulled
    m = tmp_path / "manifests" / "ingest_manifest.jsonl"
    lines = m.read_text().splitlines()
    m.write_text("\n".join(lines[:-1]) + "\n")
    ingest(["AAA"], s, e, "1Min", layers=("adjusted",), root=tmp_path, fetch=_fake_fetch(calls))
    assert len(calls) == n1 + 1
    assert read_bars("adjusted", "1Min", root=tmp_path).equals(first)
