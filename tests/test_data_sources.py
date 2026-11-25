"""Symbol lists and the Yahoo downloader's manners (no network: urlopen is faked)."""
import io
import json
import urllib.error
from pathlib import Path

import pytest

from forecast.data import sources

ROOT = Path(__file__).resolve().parents[1]


def _payload(n: int = 5) -> bytes:
    q = {k: [10.0 + i for i in range(n)] for k in ("open", "high", "low", "close", "volume")}
    res = {"timestamp": [1_700_000_000 + 86400 * i for i in range(n)],
           "indicators": {"quote": [q], "adjclose": [{"adjclose": q["close"]}]}}
    return json.dumps({"chart": {"result": [res]}}).encode()


def _http_error(code: int, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.invalid", code, "x", headers or {}, io.BytesIO(b""))


@pytest.fixture
def net(monkeypatch, tmp_path):
    """Fake network: ``net.plan[symbol]`` is a list of responses (bytes or exception), last one repeats."""
    class Net:
        plan: dict = {}
        calls: list = []
        sleeps: list = []

    def urlopen(req, timeout=0):
        sym = req.full_url.split("/chart/")[1].split("?")[0]
        Net.calls.append(sym)
        seq = Net.plan.get(sym, [_payload()])
        r = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(r, Exception):
            raise r
        return io.BytesIO(r)

    monkeypatch.setattr(sources.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sources.time, "sleep", lambda s: Net.sleeps.append(s))
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    return Net


def test_wide_universe_file_is_clean():
    syms = sources.read_symbols("configs/data/universe_wide.txt")
    raw = [ln.split("#")[0].strip() for ln in
           (ROOT / "configs/data/universe_wide.txt").read_text(encoding="utf-8").splitlines()]
    raw = [s for s in raw if s]
    assert len(raw) == len(set(raw)) == len(syms)          # no duplicates in the file itself
    assert 500 <= len(syms) <= 700
    assert "." not in "".join(syms)                        # Yahoo share classes use a dash
    for must in ("SPY", "QQQ", "XLK", "AAPL", "BRK-B", "BF-B"):
        assert must in syms
    assert syms.index("SPY") == 0                          # benchmark present; file order is stable


def test_read_symbols_rejects_junk(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("# head\nspy  # benchmark\n\nAAPL\nSPY\n", encoding="utf-8")
    assert sources.read_symbols(p) == ["SPY", "AAPL"]
    p.write_text("AAPL\nrm -rf /\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a symbol"):
        sources.read_symbols(p)


def test_resolve_symbols_merges_inline_and_file(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("AAPL\nMSFT\n", encoding="utf-8")
    assert sources.resolve_symbols({"symbols": ["SPY", "AAPL"], "symbols_file": str(p)}) == ["SPY", "AAPL", "MSFT"]
    with pytest.raises(ValueError):
        sources.resolve_symbols({})


def test_synthetic_load_accepts_symbols_file(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("SPY\nAAA\nBBB\n", encoding="utf-8")
    df = sources.load_bars({"source": "synthetic", "symbols_file": str(p), "n_bars": 50})
    assert sorted(df["symbol"].unique().to_list()) == ["AAA", "BBB", "SPY"]


def test_unknown_symbol_is_skipped_without_retries(net):
    net.plan = {"DEAD": [_http_error(404)]}
    net.calls, net.sleeps = [], []
    df = sources.yahoo(["SPY", "DEAD", "AAPL"])
    assert sorted(df["symbol"].unique().to_list()) == ["AAPL", "SPY"]
    assert net.calls.count("DEAD") == 1


def test_429_backs_off_then_succeeds(net):
    net.plan = {"SPY": [_http_error(429), _http_error(429, {"Retry-After": "40"}), _payload()]}
    net.calls, net.sleeps = [], []
    df = sources.yahoo(["SPY", "AAPL"])
    assert df["symbol"].n_unique() == 2
    assert net.sleeps[:2] == [15.0, 40.0]                  # long waits, and Retry-After is honoured


def test_persistent_429_stops_asking_and_uses_cache(net, tmp_path):
    net.plan, net.calls, net.sleeps = {}, [], []
    syms = ["SPY", "AAA", "BBB", "CCC"]
    sources.yahoo(syms)                                    # fills the cache
    net.calls.clear()
    net.plan = {"AAA": [_http_error(429)]}
    df = sources.yahoo(syms, max_age_h=0.0)                # everything is stale, AAA is rate limited
    assert df["symbol"].n_unique() == 4                    # stale copies still load
    assert "BBB" not in net.calls and "CCC" not in net.calls   # no more requests once blocked
    assert max(net.sleeps) <= 120.0
