"""Read-only prediction API for forecast-chat.

GET /health                -> ok + generated_at
GET /models                -> champion metadata (incl. honest validation stats)
GET /forecast/{symbol}     -> horizons x quantiles for return and vol, price bands, model + validation
"""
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException

from forecast.training.autotune import champions_dir
from forecast.utils import env

app = FastAPI(title="forecast-engine", version="0.1")


def _latest() -> dict:
    p = env.data_root() / "serving" / "latest.json"
    if not p.exists():
        raise HTTPException(503, "No predictions yet: run fe-score")
    return json.loads(p.read_text())


@app.get("/health")
def health() -> dict:
    try:
        return {"ok": True, "generated_at": _latest()["generated_at"]}
    except HTTPException:
        return {"ok": True, "generated_at": None}


@app.get("/models")
def models() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(champions_dir().glob("*.json"))]


@app.get("/forecast/{symbol}")
def forecast(symbol: str) -> dict:
    d = _latest()
    s = d["symbols"].get(symbol.upper())
    if s is None:
        raise HTTPException(404, f"{symbol} not in universe")
    return {"symbol": symbol.upper(), "generated_at": d["generated_at"], **s,
            "disclaimer": "Probabilistic research forecast; not financial advice."}
