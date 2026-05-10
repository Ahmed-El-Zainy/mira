"""api/routes.py – all REST endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from agent.core import AgentCore
from api.models import AnalysisRequest, JobStatus, MonitoringRequest
from storage.redis_client import RedisClient
from utils.config import get_settings

router = APIRouter()
_redis = RedisClient()
_settings = get_settings()
_agent = AgentCore()


# ── POST /analyze ─────────────────────────────────────────────────────────────
@router.post("/analyze", response_model=JobStatus, status_code=202)
async def analyze_company(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Submit a company analysis request. Returns job_id immediately."""
    job_id = str(uuid.uuid4())
    _redis.set_job_status(
        job_id,
        {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "tool_calls_used": 0,
        },
    )
    background_tasks.add_task(_agent.run_analysis, job_id, request.query)
    return JobStatus(job_id=job_id, status="queued")


# ── GET /status/{job_id} ──────────────────────────────────────────────────────
@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Poll the status (and final result) of an analysis job."""
    status = _redis.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatus(**{k: status.get(k) for k in JobStatus.model_fields})


# ── GET /logs/{job_id} ────────────────────────────────────────────────────────
@router.get("/logs/{job_id}")
async def get_job_logs(job_id: str):
    """Return structured per-tool invocation logs for a job."""
    logs = _redis.get_job_logs(job_id)
    if logs is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    tokens = _redis.get_job_token_usage(job_id)
    return {"job_id": job_id, "logs": logs, "token_usage": tokens}


# ── GET /jobs ─────────────────────────────────────────────────────────────────
@router.get("/jobs")
async def list_jobs(limit: int = 20):
    """Return the most-recently-created jobs (best-effort scan over Redis)."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="`limit` must be between 1 and 200.")
    return {"jobs": _redis.list_recent_jobs(limit=limit)}


# ── POST /monitor_start ───────────────────────────────────────────────────────
@router.post("/monitor_start")
async def start_monitoring(request: MonitoringRequest):
    """Register a ticker for background persistent monitoring."""
    _redis.register_monitored_ticker(request.ticker.upper(), request.cadence_hours)
    return {
        "status": "monitoring_started",
        "ticker": request.ticker.upper(),
        "cadence_hours": request.cadence_hours,
    }


# ── GET /monitor/list ─────────────────────────────────────────────────────────
@router.get("/monitor/list")
async def list_monitored():
    """Return all tickers currently registered for proactive monitoring."""
    tickers = _redis.get_all_monitored_tickers()
    rows = []
    for t in tickers:
        s = _redis.get_ticker_state(t) or {}
        rows.append({
            "ticker": t,
            "cadence_hours": s.get("cadence_hours"),
            "last_run": s.get("last_run"),
            "baseline_price": s.get("baseline_price"),
            "baseline_volume": s.get("baseline_volume"),
        })
    return {"monitored": rows}


# ── DELETE /monitor/{ticker} ──────────────────────────────────────────────────
@router.delete("/monitor/{ticker}")
async def stop_monitoring(ticker: str):
    """Remove a ticker from the monitored set."""
    removed = _redis.unregister_monitored_ticker(ticker.upper())
    return {"status": "removed" if removed else "not_found", "ticker": ticker.upper()}


# ── GET /config ───────────────────────────────────────────────────────────────
@router.get("/config")
async def get_config():
    """
    Return public LLM configuration so the demo can build its
    model selector directly from .env values — nothing hardcoded in HTML.

    Response shape (matches what loadConfig() in demo.html expects):
      {
        "llm_provider":       "hf" | "ollama" | "openai",
        "ollama_model":       "lfm2.5-thinking:1.2b",          # OLLAMA_MODEL
        "hf_model_id":        "Qwen/Qwen3.6-35B-A3B:deepinfra", # LLM_HF_MODEL_ID  ← LLM
        "hf_sentiment_model": "yiyanghkust/finbert-tone"        # HF_MODEL_ID      ← classification
      }
    """
    return {
        "llm_provider": _settings.llm_provider,
        "llm_fallback_provider": _settings.llm_fallback_provider or "",
        "ollama_model": _settings.ollama_model,
        "hf_model_id": _settings.llm_hf_model_id,
        "hf_sentiment_model": _settings.hf_model_id,
    }


# ── GET /health ───────────────────────────────────────────────────────────────
@router.get("/health")
async def health_check():
    redis_ok = _redis.ping()
    return {"status": "ok" if redis_ok else "degraded", "redis": redis_ok}
