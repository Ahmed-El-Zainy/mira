"""api/main.py – application factory and startup hooks."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from utils.config import get_settings

_settings = get_settings()

logging.basicConfig(
    level=_settings.log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

app = FastAPI(
    title="M.I.R.A. – Market Intelligence & Research Agent",
    description=(
        "Autonomous AI agent that monitors equity markets, "
        "performs deep research, and generates structured investment analysis reports."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def _startup():
    """Start the background monitoring service."""
    from monitoring.scheduler import MonitoringService
    svc = MonitoringService()
    svc.start()
