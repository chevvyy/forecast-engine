# Research: Which models actually work for stock number prediction?

Last reviewed: 2026-10-27. Re-check the "Foundation models" section every few months — it moves fast.

## 1. Frame the problem correctly first

The single biggest efficiency gain is choosing the right target. Architecture comes second.

### 1.1 Do not predict price

Raw price is non-stationary (its mean and variance drift). A network trained on price learns the
persistence trick — tomorrow ≈ today — and reports a low MSE that is useless. Always predict a
**transformed, roughly stationary target**:

| Target | Definition | Predictability | Use |
|---|---|---|---|
| Log return | `log(p[t+h] / p[t])` | Very low (near-random at short horizons) | Direction / sizing signal |
| Excess return | return − benchmark (e.g. SPY) return | Low, but removes market beta | Cross-sectional ranking |
| Realized volatility | sqrt(sum of squared 1-min returns over the next window) | **Much higher** (vol clusters) | Risk, position sizing, option-ish questions |
| Cross-sectional rank | rank of a stock's forward return among a universe | Low, but robust | Long/short style selection |

**Recommendation:** train two heads from day one — a **return quantile head** and a **realized-volatility
head**. Volatility gives you a model that demonstrably works (good for the forecast-chat demo); returns are
where the hard, honest research happens.

### 1.2 Predict distributions, not points

Output quantiles (e.g. 0.1, 0.25, 0.5, 0.75, 0.9) trained with pinball/quantile loss. This is what the
current top forecasting models do (TiRex outputs 9 quantiles; Chronos and Moirai are probabilistic). A point
forecast of a near-zero-mean return is almost always ≈ 0; the spread of the distribution is the information.

### 1.3 Horizons

Start with **5, 15, 30, and 60-minute** horizons on 1-minute bars, plus **next-day** on daily bars. Avoid
1-minute-ahead: bid/ask bounce and microstructure noise dominate, and anything you find will not survive
transaction costs.

---

## 2. What the evidence says (by model family)

### 2.1 Gradient-boosted trees (LightGBM, XGBoost, CatBoost)

- On daily, tabular features, boosted trees are the hardest baseline to beat, and they train in seconds.
  A 2025 study of time-series foundation models in finance found zero-shot Chronos and TimesFM
  underperformed CatBoost/LightGBM ensembles on daily excess returns.
- Practitioner consensus: deep sequence models only start to win at intraday frequency with high data
  volume (order flow, minute bars), where sequential structure matters.
- Caveat from a 2026 MNQ futures study (LSTM vs GBDT on 5-minute bars): walk-forward accuracy hovered
  around 48–55%, and feature importance shifted from fold to fold — i.e. the model fit each period's noise.
  Expect this. It is the normal state of intraday prediction.

**Verdict:** mandatory baseline and likely a permanent ensemble member. GPU-accelerated, fits trivially.

### 2.2 Compact deep models trained from scratch

| Model | Idea | Efficiency notes |
|---|---|---|
| **DLinear / NLinear** | Decompose + single linear layer | Shown to match or beat many Transformers on standard benchmarks. Sanity-check model. |
| **N-HiTS / N-BEATS** | Hierarchical MLP with interpolation | Fast, strong on multi-horizon. |
| **PatchTST** | Splits the series into patches (tokens), channel-independent Transformer | Patching divides sequence length by patch size → attention cost drops quadratically. Low compute. |
| **iTransformer** | Attention across variables instead of time | Good when cross-feature/cross-asset interaction matters. Did well on CSI 500 in TimeBridge's financial table. |
| **ModernTCN** | Large-kernel depthwise convolutions | Ranked first in a 918-experiment comparison across crypto/forex/equity indices (PatchTST second). **Caveat:** single-author preprint, not peer-reviewed. |
| **TimeMixer / TSMixer** | All-MLP mixing | Cheap; competitive. |
| **LSTM / GRU** | Recurrent | Older but still competitive on finance; weak on long contexts. |

Findings worth weighting:
- A controlled comparison of 9 architectures (918 runs) concluded that **architectural inductive bias mattered
  more than parameter count** — bigger did not mean better.
- A 2026 benchmark of pretrained and from-scratch models on single-ticker daily returns (AAPL, JPM, META,
  etc.) found that **none significantly beat the random walk** under Diebold–Mariano tests. This is the bar.

**Verdict:** PatchTST and ModernTCN are the primary deep candidates. Both are small (≤ a few million params),
train comfortably in 12 GB, and handle long lookbacks (e.g. 512–2048 minutes) efficiently.

### 2.3 Time-series foundation models (TSFMs)

| Model | Type | Size | Finance-relevant evidence |
|---|---|---|---|
| Chronos / Chronos-2 / Bolt (Amazon) | Tokenized values, T5-style / universal | 9M–700M | Weak zero-shot on returns; fine-tuning helped little. |
| TimesFM 2.5 (Google) | Decoder-only, patched | ~200M | Weak zero-shot on returns; needed fine-tuning to compete on volatility. |
| Moirai 2.0 (Salesforce) | Decoder-only quantile | small | Top-5 on GIFT-Eval; no special finance edge. |
| **TiRex** (NXAI) | **xLSTM (recurrent)** | **35M** | SOTA zero-shot on GIFT-Eval at release; >11× faster inference than TimesFM-2.0; TiRex-2 adds multivariate + **streaming** with constant per-patch cost. |
| **Kronos** (Tsinghua) | Finance-specific: tokenizes OHLCV candles, autoregressive | 4 sizes, MIT license | Pretrained on 12B+ K-line records from 45 exchanges; reports large RankIC gains vs generic TSFMs. AAAI 2026. |

Key finding: **generic TSFMs are not a shortcut for returns.** Multiple 2025–2026 papers show zero-shot
Chronos/TimesFM underperforming LightGBM/CatBoost on excess returns and failing to beat HAR for volatility.
One study of ~5,000 Chinese stocks found zero-shot Kronos ranked *last* for next-day volatility vs HAR-RV.
GIFT-Eval itself showed that an automatically discovered solution built on gradient boosting and
decomposition beat the entire May 2025 leaderboard — a reminder that general benchmarks reward simple,
well-engineered methods.

**Verdict:** use TSFMs in Phase 4 as (a) challengers, (b) frozen feature extractors whose embeddings feed
LightGBM, or (c) fine-tune targets. **TiRex** is the efficiency pick (small, recurrent, streaming-friendly
for live minute data). **Kronos** is the domain pick (understands OHLCV jointly). Neither replaces the
baseline ladder.

### 2.4 Things to skip

- **LLMs as forecasters** (Time-LLM, GPT-on-numbers): research shows limited benefit for the compute.
- **Reinforcement learning for trading**: a separate problem (policy, not forecast). Out of scope.
- **Price-level "accuracy" claims** (e.g. MAPE of 0.04%): almost always the persistence trick.

---

## 3. Efficiency techniques for numeric prediction (apply to every deep model)

1. **Reversible instance normalization (RevIN)** — normalize each input window, de-normalize outputs. Handles
   regime/scale drift; standard in PatchTST/TiRex-style models.
2. **Patching** — group 8–32 minutes per token. Cuts Transformer cost and adds local context.
3. **Quantile (pinball) loss** with monotonic quantile outputs.
4. **bf16 autocast** — both the 3060 (Ampere) and 5070 Ti (Blackwell) support bf16; roughly halves memory.
5. **`torch.compile`** once the model is stable.
6. **Continuous regression head vs. value tokenization** — Chronos/Kronos quantize values into bins. That helps
   pretraining at scale but loses tail fidelity; for from-scratch models keep a continuous head.
7. **Fixed-shape, pre-materialized windows** in memory-mapped arrays (or Parquet → Arrow) so the GPU never
   waits on Python feature code.
8. **Cross-asset pooling** — train one global model over many tickers (with ticker/sector embeddings) instead
   of one model per ticker. Far more data per parameter; this is how both TSFMs and the stronger GBDT setups work.

---

## 4. Final recommendation (the model ladder)

Each rung must beat the rung below it **out of sample, after costs, with a significance test** before the
next rung is worth building.

| Rung | Model | Target | Exit criterion |
|---|---|---|---|
| 0 | Zero return, random walk, rolling mean; HAR-RV for vol | both | Implemented and logged |
| 1 | Ridge / linear on engineered features | both | Beats rung 0 on vol |
| 2 | **LightGBM** (GPU), global over universe | both | Beats rung 1; stable feature importance across folds |
| 3 | **PatchTST** and **ModernTCN** from scratch, quantile heads | both | Beats LightGBM on CRPS or IC, DM test p < 0.05 |
| 4 | **TiRex** / **Kronos**: zero-shot → frozen embeddings → fine-tune | both | Beats rung 3 or adds value in ensemble |
| 5 | Ensemble (stacked on out-of-fold predictions) | both | Beats best single model after costs |

Expected outcome, stated plainly: volatility models will clearly beat baselines. Return models will most
likely show small or no edge at minute horizons. A rigorous "no edge" result is still a correct, publishable,
portfolio-worthy result.

---

## Sources

- Rahimikia et al., *Re(Visiting) Time Series Foundation Models in Finance* — https://www.researchgate.net/publication/397872068
- *Pretrained Time-Series Foundation Models for Financial Return Forecasting* (2026) — https://arxiv.org/html/2606.27100
- *Forecasting Realized Volatility with TSFMs* (2026) — https://arxiv.org/html/2607.05291v1
- *Foundation models do not beat simple volatility benchmarks: 5000 Chinese stocks* — https://www.sciencedirect.com/science/article/abs/pii/S1544612326011347
- Saidd, *Controlled Comparison of DL Architectures for Multi-Horizon Financial Forecasting* (preprint) — https://arxiv.org/abs/2603.16886
- *Sequential Structure in Intraday Futures Data: LSTM vs GB on MNQ* — https://arxiv.org/pdf/2605.17724
- TimeBridge (financial results table) — https://arxiv.org/pdf/2410.04442
- TiRex — https://alphaxiv.org/abs/2505.23719 ; TiRex-2 summary — https://themodelwire.com/article/tirex-2-generalizing-tirex-to-multivariate-data-and-streaming-01KWGF46832ZA6N8PJTNHD6KGF
- Kronos (AAAI 2026) — https://ojs.aaai.org/index.php/AAAI/article/view/39730 ; code — https://github.com/shiyu-coder/Kronos
- Moirai 2.0 — https://arxiv.org/pdf/2511.11698
- GIFT-Eval — https://arxiv.org/pdf/2410.10393 ; ERA results on GIFT-Eval — https://arxiv.org/pdf/2509.06503
- LSTM vs LightGBM practitioner summary — https://ariaanalyst.pro/blog/lstm-vs-lightgbm-stocks
