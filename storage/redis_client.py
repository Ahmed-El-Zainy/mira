"""storage/redis_client.py – thin wrapper around Redis for all M.I.R.A. state.

Optimisations vs. the original implementation:
  * Singleton connection pool — every `RedisClient()` instance shares one pool
    (was: each module created its own TCP pool, ~6 idle pools per worker).
  * HSET-based job status — partial updates are atomic, no read-modify-write,
    and individual fields can be fetched without deserialising the whole blob.
  * TTLs on every job/log key — prevents unbounded memory growth.
  * Pipelined writes for log batches.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

import redis
from redis.connection import ConnectionPool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_settings

_settings = get_settings()

# 7-day default TTL on all job-scoped keys (jobs, logs, tokens).
_JOB_TTL_SECONDS = 60 * 60 * 24 * 7

# ── module-level singletons ──────────────────────────────────────────────────
_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            host=_settings.redis_host,
            port=_settings.redis_port,
            password=_settings.redis_password or None,
            db=_settings.redis_db,
            decode_responses=True,
            max_connections=32,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30,
        )
    return _pool


def _key(namespace: str, *parts: str) -> str:
    return ":".join(["mira", namespace, *parts])


def _to_field(value: Any) -> str:
    """Serialise any Python value to a Redis hash field string."""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if value is None:
        return ""
    return str(value)


def _from_field(field: str, value: str) -> Any:
    """Deserialise hash field back to a Python value (best-effort)."""
    if value == "" and field in {"result", "error", "tag", "created_at"}:
        return None
    if field in {"progress", "tool_calls_used"}:
        try:
            return int(value)
        except ValueError:
            return 0
    if field in {"budget_exceeded"}:
        return value.lower() in ("true", "1")
    if field in {"result", "proactive_triggers"}:
        try:
            return json.loads(value) if value else None
        except json.JSONDecodeError:
            return None
    return value


class RedisClient:
    """Thin facade over Redis. All instances share one connection pool."""

    def __init__(self) -> None:
        self._r = redis.Redis(connection_pool=_get_pool())

    # ── jobs (HSET-based) ────────────────────────────────────────────────────
    def set_job_status(self, job_id: str, data: dict) -> None:
        key = _key("job", job_id, "status")
        mapping = {k: _to_field(v) for k, v in data.items()}
        with self._r.pipeline() as pipe:
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, _JOB_TTL_SECONDS)
            pipe.execute()

    def get_job_status(self, job_id: str) -> dict | None:
        key = _key("job", job_id, "status")
        raw = self._r.hgetall(key)
        if not raw:
            return None
        return {k: _from_field(k, v) for k, v in raw.items()}

    def update_job_status(self, job_id: str, patch: dict) -> None:
        """Atomic partial update — no read-modify-write race."""
        key = _key("job", job_id, "status")
        mapping = {k: _to_field(v) for k, v in patch.items()}
        with self._r.pipeline() as pipe:
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, _JOB_TTL_SECONDS)
            pipe.execute()

    def store_job_result(self, job_id: str, result: dict) -> None:
        result_key = _key("job", job_id, "result")
        with self._r.pipeline() as pipe:
            pipe.set(result_key, json.dumps(result), ex=_JOB_TTL_SECONDS)
            pipe.hset(_key("job", job_id, "status"), "result", json.dumps(result))
            pipe.expire(_key("job", job_id, "status"), _JOB_TTL_SECONDS)
            pipe.execute()

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

    def list_recent_jobs(self, limit: int = 20) -> list[dict]:
        """Return up to *limit* most-recently-touched jobs (best effort scan)."""
        cursor = 0
        keys: list[str] = []
        pattern = _key("job", "*", "status")
        while True:
            cursor, batch = self._r.scan(cursor=cursor, match=pattern, count=200)
            keys.extend(batch)
            if cursor == 0 or len(keys) >= limit * 4:
                break
        rows: list[dict] = []
        for k in keys[: limit * 4]:
            data = self._r.hgetall(k)
            if data:
                rows.append({kk: _from_field(kk, vv) for kk, vv in data.items()})
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
        return rows[:limit]

    # ── per-job logs ──────────────────────────────────────────────────────────
    def add_job_log(self, job_id: str, entry: dict) -> None:
        key = _key("job", job_id, "logs")
        with self._r.pipeline() as pipe:
            pipe.rpush(key, json.dumps(entry))
            pipe.expire(key, _JOB_TTL_SECONDS)
            pipe.execute()

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
        with self._r.pipeline() as pipe:
            pipe.hincrby(key, "prompt_tokens", prompt_tokens)
            pipe.hincrby(key, "completion_tokens", completion_tokens)
            pipe.hincrby(key, "total_tokens", prompt_tokens + completion_tokens)
            pipe.hincrbyfloat(key, "estimated_cost_usd", cost_usd)
            pipe.expire(key, _JOB_TTL_SECONDS)
            pipe.execute()

    def get_job_token_usage(self, job_id: str) -> dict:
        raw = self._r.hgetall(_key("job", job_id, "tokens"))
        if not raw:
            return {}
        return {
            "prompt_tokens": int(raw.get("prompt_tokens", 0)),
            "completion_tokens": int(raw.get("completion_tokens", 0)),
            "total_tokens": int(raw.get("total_tokens", 0)),
            "estimated_cost_usd": float(raw.get("estimated_cost_usd", 0.0)),
        }

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

    def unregister_monitored_ticker(self, ticker: str) -> bool:
        with self._r.pipeline() as pipe:
            pipe.srem(_key("monitor", "tickers"), ticker)
            pipe.delete(_key("monitor", ticker))
            removed, _ = pipe.execute()
        return bool(removed)

    # ── health check ──────────────────────────────────────────────────────────
    def ping(self) -> bool:
        try:
            return self._r.ping()
        except Exception:
            return False
