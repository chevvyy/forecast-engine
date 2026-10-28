# Data Sources

Prices checked against public pages in Oct 2026. Verify before paying — these change.

## 1. Provider comparison (US equities, minute resolution)

| Provider | Plan | Price | Real-time | Historical minute bars | Notes |
|---|---|---|---|---|---|
| **Alpaca** | Basic | **$0** | IEX feed via websocket (partial volume), 30 symbols | SIP history back to 2016; SIP is 15-min delayed via REST | 200 REST calls/min. Requires a (paper) account. |
| Alpaca | Algo Trader Plus | $99/mo | Full SIP websocket | Same | Natural upgrade path; matches live API. |
| Polygon → **Massive** (rebranded early 2026) | Stocks Starter / Advanced | ~$29 / $199 per month | Delayed / full real-time | Deep history, flat files on S3 | Flat files are the fastest way to bulk-download years of minute bars. |
| **Databento** | Metered | ~$100–500/mo typical, pay per byte | Yes | Exchange-direct, L1–L3 | Best for research-grade and order-book data; cheap for bulk one-off pulls, expensive for 24/5 streaming. |
| Twelve Data / Alpha Vantage | Free–paid | Low | Limited | Limited | Fine for prototyping, not primary. |

## 2. Recommendation by phase

- **Phase 1–3 (offline research):** Alpaca free tier, historical **SIP** minute bars, `adjustment=all`.
  Cost $0. Pull a universe of ~50–100 liquid names (S&P 100 or top-volume ETFs + megacaps) for 2016–present.
- **Phase 5 (live minute pulls):** Alpaca websocket. Start on free IEX for plumbing tests, but **do not
  evaluate model quality on IEX bars** — IEX volume is a small slice of the market, so volume features and
  even prices on thin names differ from SIP. Upgrade to Algo Trader Plus ($99) when going live for real.
- **If you ever need tick/L2:** Databento, priced with their query cost calculator first.

## 3. Data pitfalls (each one has a test in `tests/`)

1. **Adjustment default is `raw`.** Alpaca returns unadjusted bars unless you request `adjustment=all`
   (splits + dividends). Unadjusted data makes splits look like −75% crashes.
2. **Corporate-action errors exist.** Alpaca's forum has repeated cases of missing/incorrect splits (NVDA,
   DELL, spurious IPO dates). Mitigation: store **raw** bars + the corporate-actions table, apply adjustments
   yourself, and cross-check daily closes against a second source for your universe.
3. **Ticker changes / shared histories.** Two symbols can map to one entity across a rename. Use the `asof`
   parameter and key everything by a stable internal security ID, not the ticker string.
4. **Survivorship bias.** A universe chosen from *today's* S&P 500 excludes companies that died. For return
   research, build the universe point-in-time (membership as of each date) or at least document the bias.
5. **Session handling.** Decide explicitly: regular hours only (09:30–16:00 ET) vs extended. Start with
   regular hours. Use an exchange calendar (`exchange_calendars` / `pandas_market_calendars`) for holidays
   and half-days. Store timestamps in UTC; convert at the edges.
6. **Missing minutes.** Illiquid minutes have no bar. Do not forward-fill volume. Forward-fill price only with
   an explicit `is_filled` flag the model can see.
7. **Bar timestamp convention.** Confirm whether a bar's timestamp is its open or close time. An off-by-one
   here is a look-ahead leak.
8. **Rate limits.** 200 calls/min on the free tier: the ingestor must be resumable and idempotent (manifest of
   completed `(symbol, month)` chunks).

## 4. Universe (initial)

Start small and liquid, expand later:
- Index ETFs: SPY, QQQ, IWM, DIA (also used as market/benchmark features)
- Sector ETFs: XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLB, XLRE, XLC
- ~40–80 megacap single names
- VIX-proxy features: VIXY or computed realized vol of SPY (VIX index itself requires separate licensing)

## 5. Schema (raw bar table)

| column | type | notes |
|---|---|---|
| security_id | string | internal stable ID |
| symbol | string | as of the bar date |
| ts_utc | timestamp[ns, UTC] | bar **open** time (verify) |
| open, high, low, close | float64 | raw, unadjusted |
| volume | int64 | |
| trade_count | int32 | |
| vwap | float64 | |
| feed | string | `sip` / `iex` |
| ingested_at | timestamp | provenance |

Adjusted tables are **derived**, never overwritten in place.

## Sources

- Alpaca pricing (checked Oct 2026) — https://aistocktradingbots.com/blog/alpaca-api-pricing/
- Provider comparison — https://aifinhub.io/articles/market-data-apis-compared-2026/ ; https://geekflare.com/guides/best-stock-market-api/ ; https://bullalert.ai/blog/best-stock-market-data-apis-2026/
- Polygon → Massive rebrand — https://github.com/api-evangelist/polygon
- Alpaca adjustment / corporate-action issues — https://forum.alpaca.markets/t/historical-stock-data-very-wrong/12047 ; https://forum.alpaca.markets/t/v2-api-bars-request-adjustment-not-working/12893 ; https://forum.alpaca.markets/t/shared-market-data-between-different-tickers/18302 ; https://forum.alpaca.markets/t/wrong-historical-data-for-past-4-years/14802
