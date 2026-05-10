"""agent/core.py – main orchestrator: plan → execute → reflect → synthesise."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from agent.executor import Executor
from agent.planner import Planner
from agent.reflection import ReflectionModule
from storage.redis_client import RedisClient
from utils.config import get_settings
from utils.llm_client import clean_llm_response, get_llm_client, get_model_name, supports_json_mode
from utils.agent_logger import AgentLogger

_settings = get_settings()
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


_COMMON_WORDS = {
    "ANALYZE", "ANALYSE", "ANALYSIS", "STOCK", "STOCKS", "TICKER", "PLEASE",
    "REPORT", "ABOUT", "TELL", "SHOW", "WHAT", "WHATS", "PRICE", "QUOTE",
    "BUY", "SELL", "HOLD", "INC", "CORP", "LTD", "LLC", "AND", "OR", "THE",
    "FOR", "WITH", "FROM", "INTO", "ON", "OF", "TO", "IN", "IS", "ARE",
    "DO", "I", "ME", "MY", "YOU", "WE", "US", "AN", "A",
}


def _coerce_ticker(plan_ticker: Any, query: str) -> str:
    """Best-effort ticker extraction.

    Small fallback models (e.g. qwen2.5:0.5b) sometimes return an empty or
    junk ticker. We try in order:
      1. The planner's value if it looks like a ticker.
      2. Any ALL-CAPS short token in the original query (likely a symbol).
      3. The first non-stopword token, upper-cased.
    """
    candidate = (str(plan_ticker or "")).strip().upper()
    if _TICKER_RE.match(candidate) and candidate not in _COMMON_WORDS:
        return candidate

    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9.\-]{0,9}", query or "")
    for token in raw_tokens:
        if token.isupper() and _TICKER_RE.match(token) and token not in _COMMON_WORDS:
            return token

    for token in raw_tokens:
        token_u = token.upper()
        if _TICKER_RE.match(token_u) and token_u not in _COMMON_WORDS:
            return token_u
    return ""

_SYNTHESIS_SYSTEM = """
You are M.I.R.A., an expert AI financial analyst. Using ONLY the structured data
provided, produce a JSON investment analysis report.

Return ONLY valid JSON matching this exact schema. No markdown fences, no explanation:
{
  "company_ticker":    "STRING",
  "company_name":      "STRING",
  "analysis_summary":  "STRING – 3-5 sentence concise synthesis",
  "sentiment_score":   FLOAT (-1.0 to 1.0) -- from news_sentiment tool,
  "hf_sentiment_score": FLOAT (-1.0 to 1.0) -- from hf_sentiment tool,
  "market_snapshot": {
    "price":               NUMBER | null,
    "daily_change_pct":    NUMBER | null,
    "market_cap":          NUMBER | null,
    "pe_ratio":            NUMBER | null,
    "52w_high":            NUMBER | null,
    "52w_low":             NUMBER | null,
    "quarterly_revenues":  [NUMBER, NUMBER]
  },
  "correlation_analysis": {
    "market_correlation":  NUMBER | null,
    "sector_etf":          "STRING",
    "sector_correlation":  NUMBER | null,
    "peer_correlations":   { "TICKER": NUMBER },
    "all_correlations":    { "TICKER": NUMBER }
  },
  "key_findings":     ["STRING", "STRING", "STRING"],
  "tools_used":       ["STRING"],
  "citation_sources": ["URL"],
  "generated_at":     "ISO8601"
}

Rules:
  • sentiment_score must equal the news_sentiment.sentiment_score value.
  • hf_sentiment_score must equal the hf_sentiment.score value.
  • market_snapshot must be populated from market_data tool results.
  • key_findings must be exactly 3 actionable, specific insights.
  • Analysis summary should briefly mention if Local and HF Cloud models agree or disagree on sentiment.
  • Do NOT invent data that is missing from the provided tool results.
""".strip()


class AgentCore:
    def __init__(self) -> None:
        self._planner    = Planner()
        self._executor   = Executor()
        self._reflection = ReflectionModule()
        self._redis      = RedisClient()
        self._logger     = AgentLogger()
        self._client     = get_llm_client()
        self._model      = get_model_name()

    # ── public entry point ────────────────────────────────────────────────────
    async def run_analysis(self, job_id: str, query: str, tag: str = "") -> None:
        job_timeout = max(30, _settings.job_timeout_seconds)
        try:
            await asyncio.wait_for(self._run(job_id, query, tag), timeout=job_timeout)
        except asyncio.TimeoutError:
            self._redis.update_job_status(
                job_id,
                {
                    "status": "failed",
                    "error": f"Job exceeded {job_timeout}s wall-clock timeout.",
                    "progress": 0,
                },
            )
            logging.warning("Job %s timed out after %ds.", job_id, job_timeout)
        except asyncio.CancelledError:
            self._redis.update_job_status(
                job_id,
                {"status": "failed", "error": "Job cancelled.", "progress": 0},
            )
            raise
        except BaseException as exc:
            self._redis.update_job_status(
                job_id,
                {"status": "failed", "error": str(exc), "progress": 0},
            )
            logging.exception("Job %s failed: %s", job_id, exc)

    # ── private pipeline ──────────────────────────────────────────────────────
    async def _run(self, job_id: str, query: str, tag: str) -> None:
        # 1. Plan
        self._redis.update_job_status(job_id, {"status": "planning", "progress": 10})
        plan = await self._planner.create_initial_plan(query)

        ticker = _coerce_ticker(plan.get("ticker"), query)
        company = (plan.get("company") or ticker or query or "").strip()
        if not ticker:
            raise ValueError(
                "Could not determine a ticker symbol from your query. "
                "Try entering a US-listed symbol like AAPL, TSLA, or MSFT."
            )
        plan["ticker"] = ticker
        plan["company"] = company

        # 2. Execute initial plan
        self._redis.update_job_status(job_id, {"status": "executing", "progress": 30})
        results = await self._executor.execute_plan(plan, job_id)

        # Check for hard errors from market_data (unknown / delisted ticker)
        mkt = results.get("market_data", {})
        if "error" in mkt:
            raise ValueError(mkt["error"])

        # 3. Reflect
        self._redis.update_job_status(job_id, {"status": "reflecting", "progress": 65})
        reflection = await self._reflection.evaluate_results(results, query)

        if reflection.get("needs_more_research"):
            self._redis.update_job_status(job_id, {"status": "re-planning", "progress": 70})
            extra_plan = await self._planner.create_reflection_plan(reflection, query)
            results    = await self._executor.execute_plan(extra_plan, job_id, results)
            self._redis.update_job_status(job_id, {"progress": 85})

        # 4. Synthesise final report
        self._redis.update_job_status(job_id, {"status": "synthesising", "progress": 90})
        
        # Log sentiment results for transparency
        local_s = results.get("news_sentiment", {}).get("sentiment_score", "N/A")
        cloud_s = results.get("hf_sentiment", {}).get("score", "N/A")
        logging.info(f"Synthesising results — Local Score: {local_s}, Cloud Score: {cloud_s}")

        report = await self._synthesise(ticker, company, results, query)

        if tag:
            report["tag"] = tag
        report["reflection"] = {
            "triggers_fired": reflection.get("triggers_fired", []),
            "reasoning":      reflection.get("reasoning", ""),
        }

        # 5. Persist & complete
        self._redis.store_job_result(job_id, report)
        self._redis.update_job_status(job_id, {"status": "completed", "progress": 100})

    # ── synthesis via LLM ─────────────────────────────────────────────────────
    async def _synthesise(
        self,
        ticker: str,
        company: str,
        results: dict,
        query: str,
    ) -> dict[str, Any]:
        context = {
            "ticker":       ticker,
            "company":      company,
            "query":        query,
            "data":         results,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        kwargs = dict(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user",   "content": json.dumps(context)},
            ],
        )
        if supports_json_mode():
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self._client.chat.completions.create(**kwargs)

        # If the FallbackLLMClient routed through the fallback provider, the
        # response carries `_mira_provider` / `_mira_model`. Log against the
        # actual answering model so token-usage attribution is accurate.
        actual_model = getattr(resp, "_mira_model", self._model)
        actual_provider = getattr(resp, "_mira_provider", None)
        if getattr(resp, "_mira_failover", False):
            logging.info(
                "Synthesis used FALLBACK provider %s (%s) — primary unavailable.",
                actual_provider, actual_model,
            )

        usage = resp.usage
        if usage:
            self._logger.log_token_usage(
                job_id=ticker,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                model=actual_model,
            )

        raw    = clean_llm_response(resp.choices[0].message.content)
        report = json.loads(raw)

        # Ensure citation_sources populated from news data
        articles      = results.get("news_sentiment", {}).get("articles", [])
        existing_urls = set(report.get("citation_sources") or [])
        for a in articles:
            url = a.get("url")
            if url and url not in existing_urls:
                report.setdefault("citation_sources", []).append(url)
                existing_urls.add(url)

        report["tools_used"] = list(results.keys())
        return report
