"""storage/redis_client.py – thin wrapper around Redis for all M.I.R.A. state."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import redis

from utils.config import get_settings

_settings = get_settings()


def _key(namespace: str, *parts: str) -> str:
    return ":".join(["mira", namespace, *parts])


class RedisClient:
    def __init__(self) -> None:
        self._r = redis.Redis(
            host=_settings.redis_host,
            port=_settings.redis_port,
            password=_settings.redis_password or None,
            db=_settings.redis_db,
            decode_responses=True,
        )

    # ── jobs ─────────────────────────────────────────────────────────────────
    def set_job_status(self, job_id: str, data: dict) -> None:
        self._r.set(_key("job", job_id, "status"), json.dumps(data))

    def get_job_status(self, job_id: str) -> dict | None:
        raw = self._r.get(_key("job", job_id, "status"))
        return json.loads(raw) if raw else None

    def update_job_status(self, job_id: str, patch: dict) -> None:
        current = self.get_job_status(job_id) or {}
        current.update(patch)
        self.set_job_status(job_id, current)

    def store_job_result(self, job_id: str, result: dict) -> None:
        self._r.set(_key("job", job_id, "result"), json.dumps(result))
        self.update_job_status(job_id, {"result": result})

    def get_job_result(self, job_id: str) -> dict | None:
        raw = self._r.get(_key("job", job_id, "result"))
        return json.loads(raw) if raw else None

    def create_job(self, ticker: str, tag: str = "") -> str:
        job_id = str(uuid.uuid4())
        data = {
            "job_id": job_id,
            "ticker": ticker,
            "tag": tag,
            "status": "queued",
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.set_job_status(job_id, data)
        return job_id

    # ── per-job logs ──────────────────────────────────────────────────────────
    def add_job_log(self, job_id: str, entry: dict) -> None:
        self._r.rpush(_key("job", job_id, "logs"), json.dumps(entry))

    def get_job_logs(self, job_id: str) -> list[dict]:
        raw_list = self._r.lrange(_key("job", job_id, "logs"), 0, -1)
        return [json.loads(r) for r in raw_list]

    # ── token usage ───────────────────────────────────────────────────────────
    def update_job_token_usage(
        self,
        job_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        key = _key("job", job_id, "tokens")
        current_raw = self._r.get(key)
        current = json.loads(current_raw) if current_raw else {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        }
        current["prompt_tokens"]      += prompt_tokens
        current["completion_tokens"]  += completion_tokens
        current["total_tokens"]       += prompt_tokens + completion_tokens
        current["estimated_cost_usd"] += cost_usd
        self._r.set(key, json.dumps(current))

    def get_job_token_usage(self, job_id: str) -> dict:
        raw = self._r.get(_key("job", job_id, "tokens"))
        return json.loads(raw) if raw else {}

    # ── monitoring state ──────────────────────────────────────────────────────
    def register_monitored_ticker(self, ticker: str, cadence_hours: int) -> None:
        data = {
            "ticker": ticker,
            "cadence_hours": cadence_hours,
            "last_run": None,
            "baseline_price": None,
            "baseline_volume": None,
            "last_article_ids": [],
        }
        existing = self._r.get(_key("monitor", ticker))
        if not existing:
            self._r.set(_key("monitor", ticker), json.dumps(data))
        self._r.sadd(_key("monitor", "tickers"), ticker)

    def get_all_monitored_tickers(self) -> list[str]:
        return list(self._r.smembers(_key("monitor", "tickers")))

    def get_ticker_state(self, ticker: str) -> dict | None:
        raw = self._r.get(_key("monitor", ticker))
        return json.loads(raw) if raw else None

    def update_ticker_state(self, ticker: str, patch: dict) -> None:
        current = self.get_ticker_state(ticker) or {}
        current.update(patch)
        self._r.set(_key("monitor", ticker), json.dumps(current))

    # ── health check ──────────────────────────────────────────────────────────
    def ping(self) -> bool:
        try:
            return self._r.ping()
        except Exception:
            return False
