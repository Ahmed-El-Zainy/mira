"""api/routes.py – all REST endpoints."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from agent.core import AgentCore
from api.models import AnalysisRequest, JobStatus, MonitoringRequest
from storage.redis_client import RedisClient
from utils.config import get_settings

router     = APIRouter()
_redis     = RedisClient()
_settings  = get_settings()
_agent     = AgentCore()


# ── POST /analyze ─────────────────────────────────────────────────────────────
@router.post("/analyze", response_model=JobStatus, status_code=202)
async def analyze_company(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Submit a company analysis request. Returns job_id immediately."""
    job_id = str(uuid.uuid4())
    _redis.set_job_status(job_id, {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "tool_calls_used": 0,
    })
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


# ── POST /monitor_start ───────────────────────────────────────────────────────
@router.post("/monitor_start")
async def start_monitoring(request: MonitoringRequest):
    """Register a ticker for background persistent monitoring."""
    _redis.register_monitored_ticker(request.ticker.upper(), request.cadence_hours)
    return {
        "status":        "monitoring_started",
        "ticker":        request.ticker.upper(),
        "cadence_hours": request.cadence_hours,
    }


# ── GET /health ───────────────────────────────────────────────────────────────
@router.get("/health")
async def health_check():
    redis_ok = _redis.ping()
    return {"status": "ok" if redis_ok else "degraded", "redis": redis_ok}
