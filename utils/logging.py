"""utils/logging.py – structured logging + token/cost tracking."""
import logging
import json
from datetime import datetime, timezone
from typing import Any

from storage.redis_client import RedisClient

logger = logging.getLogger("mira")


class AgentLogger:
    # GPT-4o pricing (per 1 K tokens, USD) as of mid-2024
    _COST_TABLE: dict[str, dict[str, float]] = {
        "gpt-4o":       {"prompt": 0.005,  "completion": 0.015},
        "gpt-4":        {"prompt": 0.03,   "completion": 0.06},
        "gpt-3.5-turbo":{"prompt": 0.001,  "completion": 0.002},
    }

    def __init__(self) -> None:
        self._redis = RedisClient()

    # ── tool invocation ──────────────────────────────────────────────────────
    def log_tool_invocation(
        self,
        job_id: str,
        tool_name: str,
        inputs: dict,
        outputs: Any,
        latency_ms: float,
        success: bool,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "event_type": "tool_invocation",
            "tool_name": tool_name,
            "inputs": inputs,
            "outputs": outputs if success else None,
            "error": outputs if not success else None,
            "latency_ms": latency_ms,
            "success": success,
        }
        self._redis.add_job_log(job_id, entry)
        logger.info(json.dumps(entry))

    # ── token usage ──────────────────────────────────────────────────────────
    def log_token_usage(
        self,
        job_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> None:
        table = self._COST_TABLE.get(model, self._COST_TABLE["gpt-4o"])
        prompt_cost     = (prompt_tokens     / 1_000) * table["prompt"]
        completion_cost = (completion_tokens / 1_000) * table["completion"]
        total_cost      = prompt_cost + completion_cost

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "event_type": "token_usage",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "model": model,
            "estimated_cost_usd": round(total_cost, 6),
        }
        self._redis.add_job_log(job_id, entry)
        self._redis.update_job_token_usage(
            job_id, prompt_tokens, completion_tokens, total_cost
        )
        logger.info(json.dumps(entry))
