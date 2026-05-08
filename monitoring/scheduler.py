"""monitoring/scheduler.py – background monitoring with configurable cadence."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from threading import Thread

import schedule

from monitoring.triggers import TriggerEvaluator
from storage.redis_client import RedisClient
from utils.config import get_settings

logger = logging.getLogger("mira.monitor")
_settings = get_settings()


class MonitoringService:
    def __init__(self) -> None:
        self._redis    = RedisClient()
        self._triggers = TriggerEvaluator()
        self._running  = False

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        self._running = True
        t = Thread(target=self._run_scheduler, daemon=True)
        t.start()
        logger.info("MonitoringService started.")

    def stop(self) -> None:
        self._running = False

    # ── scheduler loop ────────────────────────────────────────────────────────
    def _run_scheduler(self) -> None:
        schedule.every(1).hours.do(self._check_all_tickers)
        while self._running:
            schedule.run_pending()
            import time; time.sleep(30)

    def _check_all_tickers(self) -> None:
        tickers = self._redis.get_all_monitored_tickers()
        for ticker in tickers:
            state    = self._redis.get_ticker_state(ticker)
            cadence  = (state or {}).get("cadence_hours", _settings.default_monitoring_cadence_hours)

            last_run_str = (state or {}).get("last_run")
            if last_run_str:
                last_run = datetime.fromisoformat(last_run_str)
                elapsed  = (datetime.now(timezone.utc) - last_run).total_seconds() / 3600
                if elapsed < cadence:
                    continue

            result = self._triggers.evaluate(ticker, state)
            if result.triggered:
                self._fire_analysis(ticker, result)
                self._redis.update_ticker_state(ticker, {
                    **result.new_state,
                    "last_run": datetime.now(timezone.utc).isoformat(),
                })

    def _fire_analysis(self, ticker: str, result: "TriggerResult") -> None:  # type: ignore[name-defined]
        from agent.core import AgentCore  # lazy import to avoid circular
        job_id = self._redis.create_job(ticker, "PROACTIVE_ALERT")
        self._redis.update_job_status(job_id, {
            "proactive_triggers": result.triggers,
            "trigger_reason":     result.reason,
        })
        query = f"Analyse {ticker} – PROACTIVE_ALERT triggered by: {result.reason}"
        asyncio.run(AgentCore().run_analysis(job_id, query, tag="PROACTIVE_ALERT"))
        logger.info("PROACTIVE_ALERT fired for %s (job %s): %s", ticker, job_id, result.reason)
