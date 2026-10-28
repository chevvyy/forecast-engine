# forecast.serving

## Files to create
| File | Responsibility |
|---|---|
| `registry.py` | Load the "production" model version from MLflow model registry |
| `batch.py` | After-close scoring → predictions table (Parquet + optional Postgres) |
| `stream.py` | (Phase 5) Incremental features on each new bar → score → publish |
| `api.py` | FastAPI: `GET /forecast/{symbol}` → horizons × quantiles for return and vol, model version, validation metrics, timestamp |
| `monitor.py` | Daily realized-vs-predicted comparison; drift alerts |

## Contract with forecast-chat
The chat should present ranges and model confidence, never a single price target. Return quantiles are
converted to price bands at display time: `price_now × exp(q)`. Include the model's validation metrics in
the response so the chat can state how good (or not) the model is.

## Latency budget (Phase 5)
Bar close → prediction available: < 5 s for the full universe.
