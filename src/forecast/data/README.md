# forecast.data

Build this first. Nothing downstream is trustworthy until these tests pass.

## What exists
`sources.py` (synthetic, keyless Yahoo, symbol lists) and `alpaca.py` (minute-bar ingestor: raw
per-symbol-month parquet, resumable, adjusted on load) and `calendar.py` (early closes only, as a table
of dates: the 210-minute session and the time of day on such a day; no holiday list, no minute grid).
The table below is the fuller plan.

## Files to create
| File | Responsibility |
|---|---|
| `ingest.py` | Historical REST pulls from Alpaca in `(symbol, month)` chunks; writes raw Parquet; manifest for resume; respects 200 calls/min |
| `corporate_actions.py` | Pull + store splits/dividends/symbol changes; never trust provider-adjusted bars blindly |
| `adjust.py` | Pure function: raw bars + actions → adjusted bars (new table, raw untouched) |
| `calendar.py` | XNYS sessions, half-days, full regular-session minute grid |
| `align.py` | Reindex to the minute grid; forward-fill price with `is_filled`; volume = 0 on gaps |
| `store.py` | Partitioned Parquet read/write (`symbol=/year=/month=`), dataset snapshot hashing |
| `quality.py` | DQ report: OHLC sanity, gaps, outlier returns, split cross-check vs second source |
| `live_ingest.py` | (Phase 5) websocket minute-bar consumer → same raw schema |

## Required tests (`tests/data/`)
- Ingest is idempotent: running twice produces identical files.
- Resume after a simulated crash mid-chunk loses and duplicates nothing.
- Adjust: synthetic 4:1 split produces a continuous adjusted series; raw unchanged.
- Timestamps are tz-aware UTC everywhere (ruff DTZ rule enforces no naive datetimes).
- Half-day sessions have 210 minutes, not 390.
- Bar timestamp convention (open vs close) asserted against a known provider sample.
