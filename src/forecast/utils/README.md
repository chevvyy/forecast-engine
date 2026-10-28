# forecast.utils

| File | Responsibility |
|---|---|
| `config.py` | Load/validate YAML configs (Hydra or pydantic models) |
| `logging.py` | Structured logging (JSON lines) shared by all modules |
| `env.py` | Read `.env`; fail fast on missing keys |
| `timeutil.py` | UTC/exchange-time conversions; no naive datetimes leave this module |
