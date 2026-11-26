"""Alpaca minute ingestor against a fake server (no network, no keys): paging, the regular-session
filter, the 15-minute rule, resume after a crash, incremental updates and split adjustment."""
import io
import json
import urllib.error
import urllib.parse
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from forecast.data import alpaca, sources
from forecast.loop import worker

NY = ZoneInfo("America/New_York")
SPLIT = datetime(2024, 2, 12, tzinfo=NY)        # 4:1 split effective this morning
START = datetime(2024, 1, 15, tzinfo=UTC)
NOW = datetime(2024, 3, 6, 15, 7, 30, tzinfo=UTC)   # a Wednesday, 10:07:30 in New York
CREDS = ("test-key-id", "test-secret")


def _raw_price(t: datetime) -> float:
    return 400.0 if t < SPLIT else 100.0


def _server_bars(sym: str, timeframe: str, adjustment: str, start: datetime, end: datetime) -> list[dict]:
    """Bars every 30 minutes from 08:00 to 17:00 New York time on weekdays (so some are outside the
    regular session), or one bar per weekday at New York midnight for 1Day."""
    out = []
    day = start.astimezone(NY).replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        if day.weekday() < 5:
            slots = [day] if timeframe == "1Day" else [day.replace(hour=8) + timedelta(minutes=30 * i)
                                                       for i in range(19)]
            for t in slots:
                if start <= t <= end:
                    c = 100.0 if adjustment == "all" else _raw_price(t)
                    out.append({"t": t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "o": c, "h": c + 1, "l": c - 1, "c": c, "v": 1000, "n": 5, "vw": c})
        day = (day + timedelta(days=1)).replace(hour=0)
    return out


@pytest.fixture
def api(monkeypatch, tmp_path):
    class Api:
        requests: list = []       # (symbol, params) per call
        fail_after: int | None = None
        errors: list = []         # exceptions to raise first, one per call
        sleeps: list = []

    Api.requests, Api.errors, Api.sleeps, Api.fail_after = [], [], [], None

    def urlopen(req, timeout=0):
        assert req.full_url.startswith("https://data.alpaca.markets/v2/stocks/")
        assert req.get_header("Apca-api-key-id") == CREDS[0]
        url = urllib.parse.urlsplit(req.full_url)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
        sym = url.path.split("/")[-2]
        if Api.errors:
            raise Api.errors.pop(0)
        if Api.fail_after is not None and len(Api.requests) >= Api.fail_after:
            raise urllib.error.HTTPError(req.full_url, 403, "down", {}, io.BytesIO(b""))
        Api.requests.append((sym, q))
        assert q["feed"] == "sip" and q["limit"] == "10000"
        s, e = (datetime.fromisoformat(q[k].replace("Z", "+00:00")) for k in ("start", "end"))
        bars = _server_bars(sym, q["timeframe"], q["adjustment"], s, e)
        i = int(q.get("page_token", 0))
        page = {"bars": bars[i:i + 40], "symbol": sym,
                "next_page_token": str(i + 40) if i + 40 < len(bars) else None}
        return io.BytesIO(json.dumps(page).encode())

    monkeypatch.setattr(alpaca.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(alpaca.time, "sleep", lambda s: Api.sleeps.append(s))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    Api.root = tmp_path / "alpaca" / "1Min"
    return Api


def _snapshot(root) -> dict:
    return {str(p.relative_to(root)): pl.read_parquet(p) for p in sorted(root.rglob("*.parquet"))}


def test_no_keys_means_off_and_heartbeat_says_so(monkeypatch, tmp_path):
    for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    env = tmp_path / ".env"
    assert alpaca.credentials(env) is None
    env.write_text("ALPACA_API_KEY=\nALPACA_SECRET_KEY=\n", encoding="utf-8")     # the .env.example state
    assert alpaca.credentials(env) is None
    env.write_text('ALPACA_API_KEY=abc  # paper\nALPACA_SECRET_KEY="xyz"\n', encoding="utf-8")
    assert alpaca.credentials(env) == ("abc", "xyz")
    monkeypatch.setattr(alpaca, "REPO", tmp_path / "nowhere")
    with pytest.raises(alpaca.MissingKeysError):
        alpaca.ingest(["SPY"], START)
    with pytest.raises(alpaca.MissingKeysError):
        sources.load_bars({"source": "alpaca", "symbols": ["SPY"], "start": "2024-01-15"})
    worker.heartbeat("w1", {"device": "cpu"}, tmp_path)
    hb = json.loads((tmp_path / "loop" / "workers" / "w1.json").read_text(encoding="utf-8"))
    assert hb["alpaca_keys"] is False
    monkeypatch.setenv("ALPACA_API_KEY", "abc")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "xyz")
    worker.heartbeat("w1", {"device": "cpu"}, tmp_path)
    text = (tmp_path / "loop" / "workers" / "w1.json").read_text(encoding="utf-8")
    assert json.loads(text)["alpaca_keys"] is True and "abc" not in text and "xyz" not in text


def test_ingest_pages_filters_session_and_respects_the_delay(api):
    stats = alpaca.ingest(["SPY"], START, NOW, CREDS)
    assert not stats["failed"]
    files = sorted(p.name for p in (api.root / "SPY").iterdir())
    assert files == ["2024-01.parquet", "2024-02.parquet", "2024-03.partial.parquet"]
    assert any("page_token" in q for _, q in api.requests)             # paging really happened
    cutoff = NOW - timedelta(minutes=15)
    for _, q in api.requests:
        assert q["adjustment"] == "raw" and q["timeframe"] == "1Min"
        assert datetime.fromisoformat(q["end"].replace("Z", "+00:00")) <= cutoff
    df = alpaca.stored("SPY")
    assert df.schema["t"] == pl.Datetime("us", "UTC")
    assert df["t"].is_sorted() and df["t"].n_unique() == df.height
    local = df["t"].dt.convert_time_zone("America/New_York")
    assert {t.strftime("%H:%M") for t in local.to_list()} == {
        f"{h:02d}:{m:02d}" for h in range(9, 16) for m in (0, 30)} - {"09:00"}
    # t is the bar's start: 09:30 is the first regular bar, 15:30 the last; 16:00 belongs to after-hours
    assert df["t"].min() == datetime(2024, 1, 15, 14, 30, tzinfo=UTC)
    assert df["t"].max() == datetime(2024, 3, 6, 14, 30, tzinfo=UTC)   # 09:30 ET; 10:00 ET is too fresh
    jan_days = df.filter(pl.col("t") < datetime(2024, 2, 1, tzinfo=UTC))["t"].dt.date().n_unique()
    assert jan_days == 13 and stats["bars"] == df.height


def test_rerun_is_idempotent_and_skips_finished_months(api):
    alpaca.ingest(["SPY", "QQQ"], START, NOW, CREDS)
    before = _snapshot(api.root)
    api.requests.clear()
    stats = alpaca.ingest(["SPY", "QQQ"], START, NOW, CREDS)
    assert stats["months_skipped"] == 4 and stats["bars"] == 0
    assert all(q["start"] >= "2024-03-06" for _, q in api.requests)    # only the open month is touched
    after = _snapshot(api.root)
    assert before.keys() == after.keys() and all(before[k].equals(after[k]) for k in before)


def test_crash_then_resume_loses_and_duplicates_nothing(api, tmp_path, monkeypatch):
    alpaca.ingest(["SPY", "QQQ"], START, NOW, CREDS)
    want = _snapshot(api.root)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "second"))
    api.requests.clear()
    api.fail_after = 3                                                  # dies in the middle of a month
    stats = alpaca.ingest(["SPY", "QQQ"], START, NOW, CREDS)
    assert stats["failed"] and not list((tmp_path / "second").rglob("*.tmp"))
    api.fail_after = None
    assert not alpaca.ingest(["SPY", "QQQ"], START, NOW, CREDS)["failed"]
    got = _snapshot(tmp_path / "second" / "alpaca" / "1Min")
    assert want.keys() == got.keys() and all(want[k].equals(got[k]) for k in want)


def test_open_month_is_extended_then_frozen(api, tmp_path, monkeypatch):
    alpaca.ingest(["SPY"], START, NOW, CREDS)
    api.requests.clear()
    alpaca.ingest(["SPY"], START, NOW + timedelta(hours=2), CREDS)
    assert [q["start"] for _, q in api.requests] == ["2024-03-06T14:31:00Z"]   # last bar + 1 minute
    assert alpaca.stored("SPY")["t"].max() == datetime(2024, 3, 6, 16, 30, tzinfo=UTC)
    alpaca.ingest(["SPY"], START, datetime(2024, 4, 3, 15, 0, tzinfo=UTC), CREDS)
    names = sorted(p.name for p in (api.root / "SPY").iterdir())
    assert names == ["2024-01.parquet", "2024-02.parquet", "2024-03.parquet", "2024-04.partial.parquet"]
    df = alpaca.stored("SPY")
    assert df["t"].n_unique() == df.height
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "one_go"))      # the same store built in one go
    alpaca.ingest(["SPY"], START, datetime(2024, 4, 3, 15, 0, tzinfo=UTC), CREDS)
    assert alpaca.stored("SPY").equals(df)


def test_split_is_adjusted_on_load_and_raw_store_is_untouched(api, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", CREDS[0])
    monkeypatch.setenv("ALPACA_SECRET_KEY", CREDS[1])
    df = alpaca.load(["SPY", "QQQ"], START, NOW)
    assert df.columns == sources.COLS and df["symbol"].n_unique() == 2
    assert df.sort(["symbol", "t"]).equals(df)
    assert df["close"].min() == pytest.approx(100.0) and df["close"].max() == pytest.approx(100.0)
    raw = alpaca.stored("SPY")
    assert sorted(raw["close"].unique().to_list()) == [100.0, 400.0]   # raw keeps the 4:1 step
    assert df.filter(pl.col("symbol") == "SPY")["t"].equals(raw["t"])
    assert df["volume"].min() == 1000.0                                # volume is not rescaled
    via_cfg = sources.load_bars({"source": "alpaca", "symbols": ["SPY", "QQQ"], "start": "2024-01-15"})
    assert via_cfg.filter(pl.col("t") <= df["t"].max()).height == df.height


def test_rate_limit_backs_off_and_bad_keys_fail_fast(api):
    def err(code):
        return urllib.error.HTTPError("https://example.invalid", code, "x", {}, io.BytesIO(b""))

    api.errors = [err(429), err(429)]
    df = alpaca.fetch_bars("SPY", datetime(2024, 3, 4, tzinfo=UTC), datetime(2024, 3, 5, tzinfo=UTC), CREDS)
    assert df.height and [s for s in api.sleeps if s >= 20] == [20.0, 40.0]
    api.errors = [err(401)]
    n = len(api.requests)
    with pytest.raises(RuntimeError, match="401") as e:
        alpaca.fetch_bars("SPY", datetime(2024, 3, 4, tzinfo=UTC), datetime(2024, 3, 5, tzinfo=UTC), CREDS)
    assert len(api.requests) == n and CREDS[1] not in str(e.value)


def test_daily_bars_come_adjusted_cached_and_auto_prefers_alpaca(api, monkeypatch):
    from forecast.data.sources import load_bars

    monkeypatch.setenv("ALPACA_API_KEY", CREDS[0])
    monkeypatch.setenv("ALPACA_SECRET_KEY", CREDS[1])
    cfg = {"source": "auto", "interval": "1d", "symbols": ["AAA", "BBB"], "start": "2016-01-04"}
    df = load_bars(cfg)
    assert set(df["symbol"].unique()) == {"AAA", "BBB"} and df.height > 100
    assert all(q["timeframe"] == "1Day" and q["adjustment"] == "all" for _, q in api.requests)
    n = len(api.requests)
    assert load_bars(cfg).equals(df) and len(api.requests) == n        # second call is served from cache
    assert df["t"].min() >= datetime(2016, 1, 4, tzinfo=UTC)


def test_auto_without_keys_uses_yahoo(monkeypatch, tmp_path):
    from forecast.data import sources

    for k in ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(alpaca, "REPO", tmp_path / "nowhere")
    seen = []
    monkeypatch.setattr(sources, "yahoo", lambda syms, interval: seen.append(interval) or sources.synthetic(syms, 80))
    assert sources.load_bars({"source": "auto", "symbols": ["SPY", "AAA"]}).height == 160 and seen == ["1d"]


def test_pinned_end_freezes_daily_bars_and_dead_symbols_are_asked_once(api, monkeypatch):
    """Wide universe (2026-11-26): with ``data.end`` every job must see the same bars however old the
    cache is, share classes go out in Alpaca's spelling, and a symbol Alpaca does not know costs one
    request in total, not a retry loop in every job."""
    from forecast.data.sources import load_bars

    monkeypatch.setenv("ALPACA_API_KEY", CREDS[0])
    monkeypatch.setenv("ALPACA_SECRET_KEY", CREDS[1])
    cfg = {"source": "auto", "interval": "1d", "symbols": ["AAA", "BRK-B", "GONE", "VOID"],
           "start": "2016-01-04", "end": "2024-03-01"}
    real = alpaca.urllib.request.urlopen

    def urlopen(req, timeout=0):
        if "/GONE/" in req.full_url:
            api.requests.append(("GONE", {}))
            raise urllib.error.HTTPError(req.full_url, 404, "no such symbol", {}, io.BytesIO(b""))
        if "/VOID/" in req.full_url:
            api.requests.append(("VOID", {}))
            return io.BytesIO(json.dumps({"bars": None, "next_page_token": None}).encode())
        return real(req, timeout)

    monkeypatch.setattr(alpaca.urllib.request, "urlopen", urlopen)
    df = load_bars(cfg)
    asked = [s for s, _ in api.requests]
    assert list(dict.fromkeys(asked)) == ["AAA", "BRK.B", "GONE", "VOID"]
    assert asked.count("GONE") == 1 and asked.count("VOID") == 1 and max(api.sleeps, default=0) < 1
    assert set(df["symbol"].unique()) == {"AAA", "BRK-B"}
    assert df["t"].max() == datetime(2024, 3, 1, 5, tzinfo=UTC)          # the bar of data.end, none later
    assert (api.root.parent / "1Day" / "GONE.none").exists()
    # Three weeks later, cache long past max_age_h: nothing is fetched and the bars are identical.
    wall = alpaca.time.time()
    monkeypatch.setattr(alpaca.time, "time", lambda: wall + 21 * 86400)
    n = len(api.requests)
    assert load_bars(cfg).equals(df) and len(api.requests) == n
    assert (api.root.parent / "1Day" / "VOID.none").exists()
    # A pin that is still in the future cannot freeze anything: the age rule applies, so it refetches.
    alpaca.daily(["AAA", "BRK-B"], START, end=datetime(2999, 1, 1, tzinfo=UTC))
    assert len(api.requests) > n


def test_end_pin_applies_to_every_source():
    from forecast.data.sources import end_exclusive, load_bars

    cfg = {"source": "synthetic", "symbols": ["SPY", "AAA"], "n_bars": 400}
    full = load_bars(cfg)
    cut = load_bars({**cfg, "end": "2015-06-30"})
    assert end_exclusive({"end": "2015-06-30"}) == datetime(2015, 7, 1, tzinfo=UTC)
    assert cut["t"].max() == datetime(2015, 6, 30, tzinfo=UTC) and cut.height < full.height
    assert cut.equals(full.filter(pl.col("t") < datetime(2015, 7, 1, tzinfo=UTC)))
