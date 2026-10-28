# ADR-0002: Python + PyTorch for modeling; Polars + Parquet + DuckDB for data

- Status: Accepted
- Date: 2026-10-27

## Context
Candidates: PyTorch, JAX, Julia, Rust (burn/candle), C++ (libtorch). Data: pandas, Polars, DuckDB, Spark.

## Decision
PyTorch ≥ 2.7 on CUDA 12.8+ wheels (required for the Blackwell RTX 5070 Ti). Polars for feature engineering,
Parquet as storage, DuckDB for SQL exploration. pandas only at library boundaries.

## Consequences
- All candidate models (PatchTST, ModernTCN, TiRex, Kronos, Chronos) are usable without porting.
- Minute-bar volume is small enough that interpreter speed is irrelevant; GPU and vectorized data code dominate.
- Revisit Rust only for a high-throughput live ingestor (trades/quotes for hundreds of symbols).
