# M.I.R.A. — Market Intelligence & Research Agent

> Autonomous AI agent that monitors equity markets, performs deep multi-step research,
> and generates structured, data-driven investment analysis reports.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                        Client                           │
└─────────────┬───────────────────────────────────────────┘
              │ REST
┌─────────────▼───────────────────────────────────────────┐
│              FastAPI  (api/main.py · api/routes.py)      │
│  POST /analyze  ·  GET /status/{id}  ·  GET /logs/{id}  │
│  POST /monitor_start  ·  GET /health                     │
└──────┬─────────────────────────────────────────┬────────┘
       │ BackgroundTask                           │ Startup
┌──────▼──────────────────────┐   ┌──────────────▼───────┐
│   AgentCore  (agent/core)   │   │  MonitoringService   │
│  ┌──────────┐ ┌──────────┐  │   │  (monitoring/        │
│  │ Planner  │ │Reflection│  │   │   scheduler.py)      │
│  └────┬─────┘ └──────────┘  │   │  TriggerEvaluator    │
│       │                     │   └──────────────────────┘
│  ┌────▼──────┐               │
│  │ Executor  │               │
│  └──┬──┬──┬──┘               │
└─────│──│──│──────────────────┘
      │  │  │
 ┌────▼┐ │ ┌▼───────────────┐
 │MktDt│ │ │PeerCorrelation │
 └─────┘ │ └────────────────┘
      ┌──▼────────────┐
      │NewsSentiment  │
      │ (FinBERT)     │
      └───────────────┘
              │
      ┌───────▼──────┐
      │   Redis      │
      │  (jobs/state │
      │   /logs)     │
      └──────────────┘
```

**Request lifecycle:**
1. `POST /analyze` → job queued in Redis, `job_id` returned immediately (async).
2. `AgentCore` runs: **Plan** → **Execute** → **Reflect** (→ **Re-plan** if needed) → **Synthesise**.
3. Final JSON report stored in Redis, retrievable via `GET /status/{job_id}`.
4. Background `MonitoringService` polls registered tickers every hour, fires `PROACTIVE_ALERT` jobs when triggers are met.

---

## Technology Choices & Rationale

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Framework | FastAPI | Native async, auto OpenAPI docs, Pydantic validation |
| LLM | OpenAI GPT-4o | Best reasoning + JSON mode; function-calling mature |
| Sentiment | FinBERT (local) | Domain-tuned for finance; zero marginal cost per article |
| Market data | yfinance | No auth required; effectively unlimited; covers price, fundamentals, OHLC |
| News | NewsAPI.org (free) | Simple REST, 100 req/day sufficient for assessment |
| State/Queue | Redis 7 | In-memory speed for job state; AOF persistence for monitoring state |
| Container | Docker + Compose | Single-command reproducible deployment |

**FinBERT vs. LLM-based sentiment trade-off:**
FinBERT runs locally, adds ~200 ms/article, needs ~500 MB of model weights on first run, but incurs zero ongoing API cost and is specifically fine-tuned on financial text (Reuters, SEC filings). LLM-based scoring (e.g., GPT-4o) would be more context-aware and handle sarcasm better, but adds ~$0.002 per article and latency. FinBERT is the right default; switch to LLM-based for edge cases like earnings-call transcripts.

---

## Setup & Run

### Prerequisites
- Docker ≥ 24 and Docker Compose V2, **or** Python 3.11+ with Redis running locally.

### Quick Start (Docker – recommended)

```bash
# 1. Clone and enter the directory
git clone <repo> && cd mira

# 2. Configure secrets
cp .env.example .env
#   → Fill in OPENAI_API_KEY and NEWS_API_KEY

# 3. Build and run
docker compose up --build

# The API is now available at http://localhost:8000
# Interactive docs: http://localhost:8000/docs
```

### Local Python (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys
# Ensure Redis is running locally (default port 6379)
uvicorn api.main:app --reload
```

### Running Tests

```bash
pytest -v
```

> Integration tests require Redis. Unit tests (e.g., `test_market_data_tool_raises_on_bad_ticker`) run without it.

---

## API Usage

### Submit an analysis
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyse the near-term prospects of Tesla, Inc. (TSLA)."}'
# → {"job_id": "abc-123", "status": "queued", "progress": 0}
```

### Poll status / get result
```bash
curl http://localhost:8000/status/abc-123
# → {"status": "completed", "progress": 100, "result": {...}}
```

### View structured logs + token usage
```bash
curl http://localhost:8000/logs/abc-123
```

### Start persistent monitoring
```bash
curl -X POST http://localhost:8000/monitor_start \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA", "cadence_hours": 24}'
```

---

## Evaluation Framework (Section 3.D)

### Test Cases (3 documented)

| # | Input | Expected behaviour |
|---|-------|--------------------|
| 1 | `"Analyse Apple Inc. (AAPL)"` | `correlation_analysis.all_correlations["AAPL"] == 1.0` (self-correlation identity) |
| 2 | `"Analyse ZZZZZ99_UNKNOWN_TICKER_XYZ"` | `status == "failed"`, error contains "not found" – no hallucinated report |
| 3 | `"Analyse Toys R Us (TOY)"` | `status == "failed"`, error indicates ticker unavailable/delisted |

### Measuring Agent Quality at Scale

Evaluating M.I.R.A. at scale requires a multi-layered approach because a single metric can mask failure modes that only surface in production.

**Ground-truth comparisons.** For a rolling weekly sample of analyses, compare M.I.R.A.'s directional sentiment prediction against the stock's actual 5-day return. A well-calibrated agent should show statistically significant correlation (p < 0.05) for positive-sentiment reports. Compare key findings against Bloomberg/FactSet analyst summaries for coverage overlap.

**LLM-as-Judge.** Deploy a separate GPT-4o instance (different temperature, different system prompt) to score each report on four rubrics scored 1–5: (a) factual grounding (are all claims traceable to a cited source?), (b) coherence and logical flow, (c) completeness against the required schema, (d) actionability of key findings. Track p50/p95 scores over time and alert on regressions.

**Regression test suite.** Maintain a curated suite of ~50 tickers spanning normal, volatile, delisted, and data-sparse cases. Run it on every model or prompt change in CI. Gate merges on zero new failures and < 5% degradation in LLM-Judge scores.

**Financial backtesting.** For historical analyses, evaluate whether acting on M.I.R.A.'s `key_findings` outperformed a buy-and-hold SPY benchmark over a 5-day window. Apply Sharpe-ratio and maximum-drawdown constraints to avoid overfitting on lucky calls.

**Operational & cost metrics.** Track p99 latency per job, tool-call budget utilisation, failure rate by failure category (timeout / bad ticker / LLM error), and estimated cost per analysis. Alert when cost > $0.50/job or failure rate > 2%.

By combining these five lenses – ground truth, judge model, regression suite, backtest, and operational metrics – we get a comprehensive and honest picture of agent quality that scales with usage volume.

---

## Known Limitations

1. **yfinance data freshness.** yfinance scrapes Yahoo Finance and can have 15-minute delays or stale fundamental data for less-liquid stocks. A premium provider (Polygon.io, Refinitiv) is recommended for production.
2. **FinBERT context window.** Input is truncated to 512 tokens (title + description). Long earnings-call transcripts will be clipped; LLM-based sentiment is better for that use case.
3. **Peer selection heuristic.** Default peers are sector-based, not company-specific. Production should use GICS sub-industry classification or competitor datasets.
4. **NewsAPI.org free tier rate limit.** 100 requests/day. High-throughput use requires a paid plan or an alternative (Marketaux, Finnhub news endpoint).
5. **Monitoring cadence granularity.** The default scheduler checks tickers hourly; intraday events between checks may be missed. A streaming approach (WebSocket feeds) is needed for sub-hourly monitoring.
6. **Single-worker concurrency.** Background tasks run in FastAPI's thread pool. Under high load, a dedicated task queue (Celery + Redis as broker, multiple workers) is required.
7. **LLM hallucination risk.** Even with strict JSON-mode prompts, the synthesis step may occasionally misstate values. All numeric claims in the report should be verified against the raw tool outputs stored in Redis.
8. **Time-zone handling.** All timestamps are UTC. Front-end or downstream consumers must handle localisation.
9. **Cost estimation accuracy.** Token cost estimates use list pricing; actual charges may differ with batch discounts, prompt caching, or model updates.
10. **No authentication layer.** The API has no auth/rate-limiting by default. Add OAuth2 or API-key middleware before any public deployment.
