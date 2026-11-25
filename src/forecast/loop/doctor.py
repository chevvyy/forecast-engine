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
    cfg = load_cfg()
    try:
        from forecast.data.sources import _yahoo_fetch, parse_yahoo_chart
        n = parse_yahoo_chart("SPY", _yahoo_fetch("SPY", cfg["data"]["interval"])).height
        print(f"[ok] market data reachable ({n} SPY bars)")
    except Exception as e:
        ok = False
        print(f"[FAIL] market data: {e}")
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
