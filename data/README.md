# data (git-ignored)

Local cache only. The source of truth is `DATA_ROOT` on the NAS, laid out as:

```
raw/bars/timeframe=1Min/symbol=AAPL/year=2024/month=03/part-0.parquet
raw/corporate_actions/...
derived/adjusted/...
derived/features/v1/<snapshot_hash>/...
manifests/ingest_manifest.parquet
```
