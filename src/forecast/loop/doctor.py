"""Pre-flight check for a worker machine. Prints one line per check and exits non-zero on a blocker.

  python -m forecast.loop.doctor
"""
from __future__ import annotations

import subprocess
import sys

from forecast.loop.run_job import load_cfg
from forecast.utils.common import REPO, device_info


def main() -> int:
    ok = True
    info = device_info()
    print(f"[{'ok' if info['device'] == 'cuda' else 'WARN'}] compute: {info}")
    if info["device"] == "cuda":
        import torch
        try:
            torch.zeros(8, device="cuda").sum().item()
            print(f"[ok] CUDA kernel runs (torch {torch.__version__}, archs {torch.cuda.get_arch_list()})")
        except Exception as e:
            ok = False
            print(f"[FAIL] CUDA present but unusable: {e}")
    try:
        # Windows has no system zone database; without the tzdata package every job dies when polars
        # hands back a UTC datetime (981 failed runs on 2026-11-25/26).
        import zoneinfo
        zoneinfo.ZoneInfo("UTC")
        zoneinfo.ZoneInfo("America/New_York")
        print("[ok] time zone database")
    except Exception as e:
        ok = False
        print(f"[FAIL] time zone database: {e!r}. Run `uv sync` (it installs tzdata) and re-run.")
    cfg = load_cfg()
    try:
        from forecast.data import alpaca, sources

        if alpaca.available():
            from datetime import UTC, datetime, timedelta
            now = datetime.now(UTC)
            n = alpaca.fetch_bars("SPY", now - timedelta(days=30), now - alpaca.DELAY, alpaca.credentials(),
                                  timeframe="1Day", adjustment="all").height
            if n == 0:
                raise RuntimeError("Alpaca answered but returned no SPY bars")
            print(f"[ok] market data reachable (Alpaca, {n} recent SPY daily bars)")
        else:
            n = sources.parse_yahoo_chart("SPY", sources._yahoo_fetch("SPY", cfg["data"]["interval"])).height
            print(f"[ok] market data reachable (Yahoo, {n} SPY bars). No Alpaca keys in .env")
    except Exception as e:
        ok = False
        print(f"[FAIL] market data: {e}")
        print("       If this is Yahoo answering 429, it is refusing scripted clients. Put Alpaca paper keys")
        print("       (ALPACA_API_KEY, ALPACA_SECRET_KEY) in a .env file in this folder and re-run.")
    r = subprocess.run(["git", "push", "--dry-run", "origin", f"HEAD:{cfg['branch']}"], cwd=REPO,
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("[ok] git push access")
    else:
        ok = False
        print(f"[FAIL] git push: {r.stderr.strip()[:300]}")
    print("READY" if ok else "NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
