<div align="center">

```
███╗   ███╗    ██╗    ██████╗    █████╗
████╗ ████║    ██║    ██╔══██╗  ██╔══██╗
██╔████╔██║    ██║    ██████╔╝  ███████║
██║╚██╔╝██║    ██║    ██╔══██╗  ██╔══██║
██║ ╚═╝ ██║    ██║    ██║  ██║  ██║  ██║
╚═╝     ╚═╝    ╚═╝    ╚═╝  ╚═╝  ╚═╝  ╚═╝
```

**Market Intelligence & Research Agent**

*Autonomous AI agent that monitors equity markets, performs deep multi-step research,*
*and generates structured, data-driven investment analysis reports.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Router-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=flat&logo=redis&logoColor=white)](https://redis.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

</div>

---

## What is M.I.R.A.?

M.I.R.A. is a production-grade agentic AI system that autonomously researches publicly traded companies and delivers structured investment analysis reports. Given a natural-language query like *"Analyse the near-term prospects of Tesla (TSLA)"*, it:

1. **Plans** a multi-step research strategy using an LLM
2. **Executes** financial data tools in sequence — market data, news sentiment, peer correlations, and cloud sentiment comparison
3. **Reflects** on result quality against concrete trigger rules, re-researching if gaps are found
4. **Synthesises** a final JSON report with actionable key findings, dual sentiment scores, and cited sources
5. **Monitors** registered tickers continuously in the background, firing proactive alerts when significant price or volume events occur

The Neural Core demo (`demo.html`) visualises the entire pipeline live — terminal logs, dual sentiment gauges, correlation heatmaps, revenue sparklines — and connects directly to the running API.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Neural Core Demo                        │
│              demo.html  (single-file frontend)              │
│   Command Bar · Dual Gauges · Terminal · Heatmap · Charts   │
└──────────────────────┬──────────────────────────────────────┘
                       │ REST / polling
┌──────────────────────▼──────────────────────────────────────┐
│         FastAPI  (api/main.py · api/routes.py)              │
│  POST /analyze · GET /status/{id} · GET /logs/{id}          │
│  POST /monitor_start · GET /health · GET /config            │
└───────┬──────────────────────────────────────────┬──────────┘
        │ BackgroundTask                            │ Startup
┌───────▼──────────────────────────┐  ┌────────────▼─────────┐
│   AgentCore  (agent/core.py)     │  │  MonitoringService   │
│                                  │  │  (monitoring/)       │
│  ┌──────────┐  ┌──────────────┐  │  │  TriggerEvaluator    │
│  │ Planner  │  │  Reflection  │  │  │  · PRICE_DEVIATION   │
│  └────┬─────┘  └──────────────┘  │  │  · VOLUME_SPIKE      │
│       │                          │  │  · NEW_ARTICLES      │
│  ┌────▼──────┐                   │  └──────────────────────┘
│  │ Executor  │                   │
│  └──┬──┬──┬──┘                   │
└─────│──│──│─────────────────────-┘
      │  │  │
      │  │  └──────────────────────── peer_correlation (yfinance)
      │  └───────────────────────────  news_sentiment  (NewsAPI + FinBERT)
      └──────────────────────────────  market_data     (yfinance)
                                        hf_sentiment   (HF Inference API)
                                               │
                                    ┌──────────▼──────────┐
                                    │        Redis        │
                                    │  jobs · logs · state│
                                    └─────────────────────┘
```

### Request Lifecycle

```
POST /analyze
    │
    ├─→ job queued in Redis → job_id returned immediately (202)
    │
    └─→ Background: AgentCore._run()
            │
            ├─ 1. Planner  → LLM generates step-by-step tool plan (JSON)
            ├─ 2. Executor → runs each tool, logs timing & outputs to Redis
            ├─ 3. Reflection → rule checks + LLM critique → re-plan if needed
            └─ 4. Synthesis → LLM writes final report JSON → stored in Redis

GET /status/{job_id}  → poll until status == "completed" | "failed"
```

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Framework | FastAPI | Native async, auto OpenAPI docs, Pydantic validation |
| LLM (primary) | HuggingFace Router | OpenAI-compatible API — swap any HF model via `.env` |
| LLM (alt) | Ollama (local) | Zero-cost, offline-capable; same OpenAI-compatible interface |
| LLM (alt) | OpenAI GPT-4o | Best JSON-mode support when budget allows |
| Sentiment (local) | FinBERT (`ProsusAI/finbert`) | Finance-domain fine-tuned; zero marginal cost per article |
| Sentiment (cloud) | `yiyanghkust/finbert-tone` via HF API | Independent second opinion; enables MATCH/DIVERGE consensus |
| Market data | yfinance | No auth; price, fundamentals, OHLC, 1Y history |
| News | NewsAPI.org | Simple REST, 100 req/day free tier |
| State / Queue | Redis 7 | In-memory job state, logs, monitoring baseline |
| Container | Docker + Compose | Single-command reproducible deployment |
| Demo UI | Vanilla HTML/JS | Single-file, zero build step; Tailwind CDN + Chart.js |

---

## Features

### Agent Pipeline
- **Planner** — chain-of-thought LLM reasoning produces a JSON execution plan per query
- **Executor** — runs `market_data`, `news_sentiment`, `peer_correlation`, `hf_sentiment` with per-tool timing and budget enforcement (`MAX_TOOL_CALLS_PER_JOB`)
- **Reflection** — three concrete trigger rules checked before synthesis:
  - `SECTOR_LOCK` — sector correlation > 0.95 (macro-dominated, not idiosyncratic)
  - `STALE_NEWS` — articles older than 72h or zero articles found
  - `NEUTRAL_SENT` — |sentiment| < 0.05 (flat signal, needs more context)
- **Synthesis** — LLM writes final report; all numeric claims traced to tool outputs

### Dual-Sentiment Engine
Two independent models score the same news stream:

| Model | Source | Type |
|-------|--------|------|
| `ProsusAI/finbert` | Local (your machine) | Classification pipeline |
| `yiyanghkust/finbert-tone` | HuggingFace Inference API | REST classification |

The demo shows both scores side-by-side with a **MATCH** (green) or **DIVERGE** (amber) consensus badge — convergence across models increases signal confidence.

### Neural Core Demo
- **Command Bar** — Spotlight-style floating input; type any ticker and press Enter
- **Live Mode** — connects to `localhost:8000`, streams real agent logs to terminal
- **Demo Mode** — fully animated mock run, works with no backend
- **Provider Switcher** — toggle Ollama ↔ HuggingFace; selector populated from `.env` via `GET /config`
- **Correlation Heatmap** — colour-coded peer correlation grid (green → amber → red)
- **Revenue Sparkline** — quarterly revenue bar chart with QoQ delta
- **Reflection Badges** — triggered quality checks shown inline with the report

### Background Monitoring
Register any ticker for continuous monitoring:
```bash
curl -X POST http://localhost:8000/monitor_start \
  -d '{"ticker": "TSLA", "cadence_hours": 24}'
```
M.I.R.A. checks every hour and fires a `PROACTIVE_ALERT` analysis job when:
- Price deviates more than 2σ from the 30-day mean
- Volume exceeds 2× the 30-day baseline
- No article baseline has been established yet

---

## Project Structure

```
mira/
├── api/
│   ├── main.py          # FastAPI app factory, CORS, startup hooks
│   ├── routes.py        # All REST endpoints incl. GET /config
│   └── models.py        # Pydantic request/response models
│
├── agent/
│   ├── core.py          # AgentCore: Plan → Execute → Reflect → Synthesise
│   ├── planner.py       # LLM-powered JSON plan generation
│   ├── executor.py      # Tool dispatcher with budget enforcement + logging
│   └── reflection.py    # Rule-based + LLM quality evaluation
│
├── tools/
│   ├── market_data.py       # yfinance: price, fundamentals, 52W range, revenue
│   ├── news_sentiment.py    # NewsAPI + local FinBERT pipeline
│   ├── peer_correlation.py  # Pearson correlations vs S&P500, sector ETF, peers
│   └── hf_sentiment.py      # HuggingFace Inference API sentiment (finbert-tone)
│
├── monitoring/
│   ├── scheduler.py     # Background thread, hourly cadence
│   └── triggers.py      # PRICE_DEVIATION / VOLUME_SPIKE / NEW_ARTICLES rules
│
├── storage/
│   ├── redis_client.py  # Job state, logs, token usage, monitoring state
│   └── models.py        # JobLog Pydantic model
│
├── utils/
│   ├── config.py        # Pydantic-settings; all .env fields in one place
│   ├── llm_client.py    # Provider router: HF / Ollama / OpenAI → AsyncOpenAI
│   └── agent_logger.py  # Structured JSON logs + token cost tracking
│
├── tests/
│   └── test_agent.py    # 4 test cases (3 integration + 1 unit)
│
├── demo.html            # Neural Core demo (single-file, no build step)
├── sample_output.json   # Example completed report
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Setup & Run

### Prerequisites

- Python 3.11+ **or** Docker ≥ 24 + Docker Compose V2
- Redis 7 (included in Docker Compose; or `brew install redis` locally)
- A HuggingFace account and token for cloud inference *(free tier works)*

### 1 — Clone & Configure

```bash
git clone https://github.com/Ahmed-El-Zainy/mira.git
cd mira
cp .env.example .env
```

Open `.env` and fill in your keys:

```bash
# ── LLM Provider (pick one) ────────────────────────────────────────────────────
LLM_PROVIDER=hf                   # "hf" | "ollama" | "openai"

# HuggingFace (recommended — free tier)
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx
LLM_HF_MODEL_ID=Qwen/Qwen3.6-35B-A3B:deepinfra   # any HF Router-compatible model
HF_BASE_URL=https://router.huggingface.co/v1

# Ollama (local, zero cost)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=lfm2.5-thinking:1.2b

# OpenAI (best JSON-mode support)
OPENAI_API_KEY=sk-xxxxxxxxxxxx
LLM_MODEL=gpt-4o

# ── Sentiment (HF classification model — separate from the LLM) ────────────────
HF_MODEL_ID=yiyanghkust/finbert-tone

# ── Data sources ───────────────────────────────────────────────────────────────
NEWS_API_KEY=xxxxxxxxxxxx          # newsapi.org free tier

# ── Redis ──────────────────────────────────────────────────────────────────────
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 2A — Docker (recommended)

```bash
docker compose up --build
```

Open [http://localhost:8000/](http://localhost:8000/) — the **Neural Core demo
loads directly** at the root path (no `/demo.html` suffix needed). The same
HTML is also served at `/demo`. Swagger docs live at `/docs`.

### 2B — Local Python

```bash
# Create virtualenv
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Fix NumPy compatibility (if you see "ARRAY_API not found")
pip install "numpy<2"

# Start Redis (if not already running)
redis-server --daemonize yes

# Start the API (demo loads at http://localhost:8000/ directly)
uvicorn api.main:app --reload
# or, equivalently, via the CLI:
mira serve --reload --open
```

### 3 — Verify

```bash
# Health check
curl http://localhost:8000/health
# → {"status": "ok", "redis": true}

# Config (confirms .env is loaded correctly)
curl http://localhost:8000/config
# → {"llm_provider": "hf", "ollama_model": "...", "hf_model_id": "Qwen/...", ...}

# Run an analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyse Alphabet Inc. (GOOGL)"}'
# → {"job_id": "abc-123", "status": "queued", "progress": 0}
```

---

## CLI

M.I.R.A. ships with a first-class command-line tool — `mira` — that talks to
either a running API server (default) **or** runs the agent in-process via
`--local` (no HTTP round-trips, no separate server).

### Install

```bash
# from the repo root
pip install -e .
# the `mira` command is now on your PATH
mira --help
```

You can also run it without installing:

```bash
python cli.py --help
```

### Common commands

```bash
# Boot the API server and pop the demo open in a browser tab
mira serve --reload --open       # http://localhost:8000/  ← Neural Core demo

# Open the demo against an already-running server
mira ui                          # opens http://localhost:8000/
mira --api http://prod:8000 ui   # against a remote deployment

# Run a one-shot analysis against a running API and stream progress
mira analyze "Analyse Tesla, Inc. (TSLA)"

# Same thing, but without an API server (Redis still required)
mira analyze "Analyse Apple (AAPL)" --local --save aapl.json

# Submit-and-forget (returns the job_id immediately)
mira analyze "Analyse NVDA" --no-watch

# Inspect a job
mira status <job_id>
mira logs   <job_id> --follow

# Persistent monitoring (works in both --local and HTTP mode)
mira monitor add TSLA --cadence 12
mira monitor list
mira monitor remove TSLA

# Health & config
mira health
mira config

# Recent jobs
mira jobs --limit 20
```

### Useful flags

| Flag          | Purpose                                                                  |
|---------------|--------------------------------------------------------------------------|
| `--api URL`   | Point at a remote M.I.R.A. server (env: `MIRA_API_URL`).                 |
| `--local`     | Run AgentCore in-process — fastest path, ideal for scripts and CI.       |
| `--json`      | Emit raw JSON instead of pretty tables — pipe into `jq`, `fx`, etc.      |
| `--quiet/-q`  | Suppress decorative output — reports collapse to a single summary line.  |

Examples piping JSON output:

```bash
mira analyze "Analyse Microsoft (MSFT)" --local --json | jq '.key_findings'
mira logs <job_id> --json | jq '.logs[].latency_ms'
```

---

## API Reference

### `POST /analyze`
Submit a natural-language analysis query. Returns a `job_id` immediately; processing runs in the background.

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyse the near-term prospects of Tesla, Inc. (TSLA)."}'
```
```json
{"job_id": "550e8400-...", "status": "queued", "progress": 0, "tool_calls_used": 0}
```

### `GET /status/{job_id}`
Poll for progress and retrieve the completed report.

```bash
curl http://localhost:8000/status/550e8400-...
```
```json
{
  "status": "completed",
  "progress": 100,
  "result": {
    "company_ticker": "TSLA",
    "sentiment_score": 0.21,
    "hf_sentiment_score": 0.18,
    "market_snapshot": { "price": 245.67, "pe_ratio": 65.4, ... },
    "correlation_analysis": { "sector_etf": "XLY", "sector_correlation": 0.89, ... },
    "key_findings": ["...", "...", "..."],
    "reflection": { "triggers_fired": [], "reasoning": "All checks passed." }
  }
}
```

**Status values:** `queued` → `planning` → `executing` → `reflecting` → `re-planning` → `synthesising` → `completed` | `failed`

### `GET /logs/{job_id}`
Structured per-tool logs with timing and token cost.

```bash
curl http://localhost:8000/logs/550e8400-...
```
```json
{
  "logs": [
    {"tool_name": "market_data", "latency_ms": 3307, "success": true, ...},
    {"tool_name": "news_sentiment", "latency_ms": 3228, "success": true, ...},
    {"tool_name": "peer_correlation", "latency_ms": 408, "success": true, ...}
  ],
  "token_usage": {"total_tokens": 6843, "estimated_cost_usd": 0.079}
}
```

### `GET /config`
Returns active LLM configuration from `.env` — used by the demo to populate the model selector without any hardcoded model IDs.

```bash
curl http://localhost:8000/config
```
```json
{
  "llm_provider": "hf",
  "ollama_model": "lfm2.5-thinking:1.2b",
  "hf_model_id": "Qwen/Qwen3.6-35B-A3B:deepinfra",
  "hf_sentiment_model": "yiyanghkust/finbert-tone"
}
```

### `POST /monitor_start`
Register a ticker for persistent background monitoring.

```bash
curl -X POST http://localhost:8000/monitor_start \
  -H "Content-Type: application/json" \
  -d '{"ticker": "TSLA", "cadence_hours": 24}'
```

### `GET /health`
```bash
curl http://localhost:8000/health
# → {"status": "ok", "redis": true}
```

Interactive docs at `http://localhost:8000/docs` (Swagger UI).

---

## Report Schema

Every completed analysis returns this JSON structure:

```json
{
  "company_ticker":    "GOOGL",
  "company_name":      "Alphabet Inc.",
  "analysis_summary":  "3–5 sentence synthesis...",

  "sentiment_score":    0.0,    // Local FinBERT  (-1.0 → 1.0)
  "hf_sentiment_score": 0.18,   // HF Cloud model (-1.0 → 1.0)

  "market_snapshot": {
    "price":              400.80,
    "daily_change_pct":   0.71,
    "market_cap":         4855869472768,
    "pe_ratio":           30.57,
    "52w_high":           402.00,
    "52w_low":            151.67,
    "quarterly_revenues": [109896000000, 113829000000]
  },

  "correlation_analysis": {
    "market_correlation":  0.5493,
    "sector_etf":          "XLC",
    "sector_correlation":  0.5489,
    "peer_correlations":   {"META": 0.2274, "NFLX": 0.0854},
    "all_correlations":    {"GOOGL": 1.0, "^GSPC": 0.5493, ...}
  },

  "key_findings":     ["...", "...", "..."],
  "tools_used":       ["market_data", "news_sentiment", "peer_correlation", "hf_sentiment"],
  "citation_sources": ["https://...", "https://..."],
  "generated_at":     "2026-05-09T18:37:57Z",

  "reflection": {
    "triggers_fired": ["STALE_NEWS", "NEUTRAL_SENT"],
    "reasoning":      "Re-research triggered due to flat sentiment and stale articles."
  }
}
```

---

## LLM Provider Configuration

M.I.R.A. uses a single `AsyncOpenAI`-compatible client for all three providers. Switch by changing `LLM_PROVIDER` in `.env` — no code changes required.

### HuggingFace (default)

```bash
LLM_PROVIDER=hf
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxx
LLM_HF_MODEL_ID=Qwen/Qwen3.6-35B-A3B:deepinfra
HF_BASE_URL=https://router.huggingface.co/v1
```

Recommended HF Router models for financial reasoning (tested):

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen3.6-35B-A3B:deepinfra` | 35B MoE | Best JSON quality; used in production logs |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Fast, reliable JSON |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 8×7B MoE | Strong reasoning, higher cost |
| `HuggingFaceH4/zephyr-7b-beta` | 7B | Good instruction following |

### Ollama (local, zero cost)

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=lfm2.5-thinking:1.2b
```

Recommended local models:

| Model | Size | Notes |
|-------|------|-------|
| `lfm2.5-thinking:1.2b` | 1.2B | Lightest; chain-of-thought reasoning |
| `qwen3.5:4b` | 3.4GB | Best quality/size ratio |
| `qwen3.5:latest` | 6.6GB | Best local reasoning + JSON |
| `gemma4:e2b` | 7.2GB | Good alternative |

> **Avoid** `ministral-3:3b`, `granite4:3b` — too weak for structured planning. Avoid `qwen2.5-coder:*` — code-focused, poor at financial reasoning.

### OpenAI

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxxxxxxxxx
LLM_MODEL=gpt-4o
```

---

## Known Issues & Fixes

### NumPy 2.x Compatibility

If you see `_ARRAY_API not found` in logs, your system NumPy (2.x) is incompatible with the compiled extensions in `torch`, `scipy`, and `transformers`:

```bash
pip install "numpy<2"
```

This is a known upstream issue. The `news_sentiment` tool gracefully catches the error and falls back to neutral scores (0.0) so the pipeline continues — you will see this warning in logs:

```
Local FinBERT failed: numpy.core.multiarray failed to import. Falling back to neutral scores.
```

The HuggingFace cloud sentiment (`hf_sentiment`) is unaffected.

### FinBERT First Run

The local FinBERT model (`ProsusAI/finbert`) is ~500MB and downloads on first use. Subsequent runs use the cached model. Set `HF_HOME` to control the cache location:

```bash
export HF_HOME=/path/to/fast/disk/.cache/huggingface
```

---

## Evaluation Framework

### Test Suite

Four tests cover the critical failure modes:

```bash
pytest -v
```

| # | Test | Input | Expected |
|---|------|-------|----------|
| 1 | Self-correlation identity | `"Analyse Apple Inc. (AAPL)"` | `all_correlations["AAPL"] == 1.0` |
| 2 | Unknown ticker → graceful error | `"Analyse ZZZZZ99_UNKNOWN"` | `status == "failed"`, error contains "not found" |
| 3 | Delisted ticker → graceful failure | `"Analyse Toys R Us (TOY)"` | `status == "failed"`, no hallucinated report |
| 4 | Unit — bad ticker raises `ValueError` | `MarketDataTool.execute("ZZZZZ99")` | `ValueError` with "not found" message |

Tests 1–3 require Redis. Test 4 runs standalone.

### Measuring Quality at Scale

Five complementary lenses for production quality assessment:

**Ground-truth comparison** — compare M.I.R.A.'s directional sentiment prediction against the stock's actual 5-day return on a rolling weekly sample. A well-calibrated agent should show statistically significant correlation (p < 0.05) for positive-sentiment reports.

**LLM-as-Judge** — deploy a separate model instance to score each report 1–5 on: factual grounding (claims traceable to cited sources), coherence, schema completeness, and actionability of key findings. Track p50/p95 over time and alert on regressions.

**Regression suite** — maintain ~50 curated tickers spanning normal, volatile, delisted, and data-sparse cases. Run on every prompt or model change in CI. Gate merges on zero new failures.

**Financial backtesting** — for historical analyses, evaluate whether acting on `key_findings` outperformed buy-and-hold SPY over a 5-day window. Apply Sharpe ratio and maximum-drawdown constraints.

**Operational metrics** — track p99 latency per job, tool-call budget utilisation, failure rate by category (timeout / bad ticker / LLM error), and cost per analysis. Alert when cost > $0.50/job or failure rate > 2%.

---

## Performance Notes & Roadmap

### Already applied (this revision)

| Area                         | Change                                                                                                  |
|------------------------------|---------------------------------------------------------------------------------------------------------|
| Tool execution               | Independent tools (`market_data`, `news_sentiment`, `hf_sentiment`) now fan out via `asyncio.gather`.    |
| Redis I/O                    | Single shared `ConnectionPool` (was: one pool per module). Job status uses `HSET` instead of GET+SET.    |
| Job/log retention            | 7-day TTLs on every job/log/token key — no more unbounded memory growth.                                |
| HuggingFace inference        | One reusable `httpx.AsyncClient`; exponential backoff on 503 / 429 (cold starts no longer return 0.0).  |
| FinBERT                      | Single batched pipeline call instead of N sequential — ~3-5× faster on 5-article default.               |
| News fetch                   | Async `httpx` (was blocking `requests`); 5-min in-process TTL cache per `(company, ticker)`.            |
| Market data                  | 60 s in-process cache; modern `asyncio.to_thread`; safer try/except around `yfinance.info`.             |
| Token usage                  | `HINCRBY` / `HINCRBYFLOAT` — atomic counters, no read-modify-write race.                                |

### Suggested next steps (ranked by impact)

1. **Move BackgroundTasks → real queue.** FastAPI's `BackgroundTasks` shares the request thread pool. Use **arq** or **Celery + Redis broker** so analyses don't compete with HTTP serving and survive restarts.
2. **Stream progress via SSE / WebSockets.** The demo currently polls `/status` every ~1 s. A `GET /stream/{job_id}` SSE endpoint that publishes Redis pub/sub events would cut perceived latency from seconds to ~10 ms.
3. **Cache LLM responses by `(prompt_hash, model)`.** Planner + reflection prompts are highly repetitive across the same ticker; an LRU on `chat.completions.create` would save 30-60% of tokens on hot paths.
4. **Switch yfinance → Polygon.io / Finnhub.** `yfinance` scrapes Yahoo and is rate-limited and brittle. A real provider gives sub-second, real-time data and proper SLAs.
5. **Persist reports to SQLite/Postgres.** Redis is great for hot state; long-term audit/backtesting needs a relational store with timeseries indexes.
6. **Trigger evaluator → batch yfinance call.** `monitoring/triggers.py` calls `yf.Ticker(t).history(...)` per ticker; `yf.download(list_of_tickers, ...)` returns all at once.
7. **APScheduler async loop** instead of `schedule` + `time.sleep(30)` in a thread — same code, no busy loop, no GIL contention.
8. **Replace deprecated `@app.on_event` with `lifespan`** (FastAPI 0.93+).
9. **Multi-stage Dockerfile.** Current image bundles `build-essential` and 4-5 GB of torch wheels at runtime. A wheel-builder stage + slim runtime would cut image size ~60%.
10. **Auth + rate limiting.** No auth today. Add API-key middleware and `slowapi` before any public deployment.
11. **Structured tracing.** Add `opentelemetry-instrumentation-fastapi` + Redis instrumentation — lets you visualise per-tool latency in Jaeger/Tempo.
12. **CI (GitHub Actions).** `pytest -v` + `ruff check` on every PR; gate merges on the regression suite passing.
13. **Pin `numpy<2` in `requirements.txt`.** README acknowledges the bug — pinning prevents anyone from hitting it.
14. **Replace `requests` with `httpx`** everywhere (one library, async-native).
15. **Job cancellation API.** `DELETE /jobs/{id}` to cancel an in-flight analysis (useful when a user closes the demo tab).

---

## Known Limitations

1. **yfinance data freshness** — 15-minute delay; stale fundamentals on illiquid stocks. Use Polygon.io or Refinitiv for production.
2. **FinBERT context window** — input truncated to 512 tokens. Long earnings transcripts get clipped; use LLM-based sentiment for those.
3. **Peer selection heuristic** — sector-based defaults, not company-specific. Use GICS sub-industry or a competitor dataset for accuracy.
4. **NewsAPI free tier** — 100 requests/day. High-throughput use needs a paid plan or Finnhub/Marketaux.
5. **NumPy 2.x incompatibility** — local FinBERT fails silently; pin `numpy<2` until upstream wheels are rebuilt.
6. **Single-worker concurrency** — background tasks share FastAPI's thread pool. Use Celery + Redis broker for production scale.
7. **LLM hallucination risk** — even with strict prompts, numeric synthesis may drift. All values in the report should be verified against raw tool outputs in Redis.
8. **No authentication** — API has no auth or rate-limiting. Add OAuth2 or API-key middleware before any public deployment.
9. **Monitoring granularity** — hourly checks miss intraday events. A WebSocket streaming approach is needed for sub-hourly monitoring.
10. **Cost estimation** — uses list pricing; actual charges differ with batch discounts or prompt caching.

---

## Contributing

```bash
# Fork → branch → commit → PR

# Run tests before submitting
pytest -v

# Check formatting
ruff check .
```

Please open an issue before submitting large changes.

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built with FastAPI · HuggingFace · FinBERT · yfinance · Redis · Chart.js

*M.I.R.A. is for research and educational purposes only.*
*Nothing in this project constitutes financial advice.*

</div>
