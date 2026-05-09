"""api/main.py – application factory and startup hooks."""
from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of the working directory.
# This runs before any other module imports settings, so all env vars
# are populated when pydantic-settings reads them.
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)  # override=True: .env vars win

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

from fastapi.responses import FileResponse

app.include_router(router)


@app.get("/", include_in_schema=False)
async def serve_demo():
    """Serve the HTML demo at the root path."""
    demo_path = Path(__file__).resolve().parent.parent / "demo.html"
    return FileResponse(demo_path)


@app.on_event("startup")
async def _startup():
    """Start the background monitoring service and verify LLM readiness."""
    from utils.llm_client import ensure_model_ready, get_model_name
    model = get_model_name()
    
    # Auto-pull model if using Ollama and it's missing
    ready = await ensure_model_ready()
    if ready:
        logging.info(f"LLM Provider Ready: {model} initialized successfully.")
    else:
        logging.error(f"LLM Readiness Check Failed for {model}.")
        logging.warning("Proceeding with caution. Agent may fail on first analysis.")

    from monitoring.scheduler import MonitoringService
    svc = MonitoringService()
    svc.start()
