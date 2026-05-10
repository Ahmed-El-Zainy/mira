"""api/main.py – application factory and startup hooks."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root regardless of the working directory.
# This runs before any other module imports settings, so all env vars
# are populated when pydantic-settings reads them.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response
from fastapi.responses import FileResponse, RedirectResponse

from api.routes import router
from utils.config import get_settings

_settings = get_settings()

logging.basicConfig(
    level=_settings.log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)


# ── lifespan replaces the deprecated @app.on_event hooks ────────────────────
@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: warm LLM (+ fallback), start monitoring. Shutdown: close clients."""
    from utils.llm_client import ensure_fallback_ready, ensure_model_ready, get_model_name
    model = get_model_name()

    ready = await ensure_model_ready()
    if ready:
        logging.info("LLM Provider Ready: %s initialized successfully.", model)
    else:
        logging.error("LLM Readiness Check Failed for %s.", model)
        logging.warning(
            "Proceeding with caution. Agent will attempt the configured "
            "fallback provider when the primary fails."
        )

    # Best-effort pre-warm of the fallback (e.g. Ollama) so the first failover
    # doesn't pay a cold-start penalty.
    await ensure_fallback_ready()

    # Pre-warm FinBERT in the background so the first /analyze isn't blocked
    # on a ~500MB model download (transformers caches it on disk afterwards).
    finbert_task = None
    if _settings.preload_finbert:
        import asyncio as _asyncio
        from tools.news_sentiment import preload_finbert
        finbert_task = _asyncio.create_task(preload_finbert(), name="preload_finbert")
        logging.info("FinBERT pre-warm scheduled in background.")

    from monitoring.scheduler import MonitoringService
    svc = MonitoringService()
    svc.start()

    try:
        yield
    finally:
        if finbert_task and not finbert_task.done():
            finbert_task.cancel()
        try:
            from tools.hf_sentiment import aclose_client
            await aclose_client()
        except Exception as exc:
            logging.warning("Error closing shared HF client: %s", exc)


app = FastAPI(
    title="M.I.R.A. – Market Intelligence & Research Agent",
    description=(
        "Autonomous AI agent that monitors equity markets, "
        "performs deep research, and generates structured investment analysis reports."
    ),
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ── Demo UI served at the root path ─────────────────────────────────────────
_DEMO_PATH = _PROJECT_ROOT / "demo.html"
_SAMPLE_PATH = _PROJECT_ROOT / "sample_output.json"

# `no-store` ensures users on the demo always pick up the latest UI changes
# without having to do a hard reload (Tailwind/Chart.js are still cached
# normally because they come from CDNs).
_NO_CACHE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}


@app.get("/", include_in_schema=False)
async def serve_demo():
    """Serve the Neural Core demo HTML at the root path."""
    if not _DEMO_PATH.exists():
        return RedirectResponse(url="/docs")
    return FileResponse(_DEMO_PATH, media_type="text/html", headers=_NO_CACHE)


@app.get("/demo", include_in_schema=False)
async def serve_demo_alias():
    """Alias so `/demo` also works."""
    return await serve_demo()


@app.get("/sample_output.json", include_in_schema=False)
async def serve_sample_output():
    """Expose the bundled sample report so the demo can render an example offline."""
    if not _SAMPLE_PATH.exists():
        return RedirectResponse(url="/")
    return FileResponse(_SAMPLE_PATH, media_type="application/json", headers=_NO_CACHE)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Avoid noisy 404s in the access log when browsers request a favicon."""
    return Response(status_code=204)
