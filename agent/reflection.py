"""agent/reflection.py – critique + self-correction with concrete trigger rules."""
from __future__ import annotations

import json

from openai import AsyncOpenAI
from utils.config import get_settings

_settings = get_settings()

_SYSTEM = """
You are M.I.R.A.'s internal quality evaluator. Analyse the tool results against
the three concrete trigger rules below and determine if a second research pass is
needed before the final report is written.

Trigger rules (check all three):
1. SECTOR_LOCK   – if sector_etf_correlation > 0.95, the ticker is moving with its
   sector, not on idiosyncratic news. Recommend fetching a direct competitor's news
   and price action.
2. STALE_NEWS    – if oldest_article_hours > 72 OR article_count == 0, news is stale.
   Recommend broadening the search or fetching SEC EDGAR filings.
3. NEUTRAL_SENT  – if |sentiment_score| < 0.05 (virtually flat), more context is
   needed. Recommend fetching analyst commentary or sector news.

Return ONLY valid JSON:
{
  "needs_more_research": true | false,
  "triggers_fired": ["SECTOR_LOCK" | "STALE_NEWS" | "NEUTRAL_SENT"],
  "reasoning": "...",
  "recommended_actions": ["action1", ...]
}
""".strip()


class ReflectionModule:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=_settings.openai_api_key)

    async def evaluate_results(self, results: dict, query: str) -> dict:
        """Apply rule-based pre-checks, then call LLM for nuanced evaluation."""
        # ── fast rule-based checks (no LLM cost) ─────────────────────────────
        triggers: list[str] = []

        corr_data = results.get("peer_correlation", {})
        sect_corr = corr_data.get("sector_correlation")
        if sect_corr is not None and sect_corr > 0.95:
            triggers.append("SECTOR_LOCK")

        news_data    = results.get("news_sentiment", {})
        oldest_hours = news_data.get("oldest_article_hours")
        art_count    = news_data.get("article_count", 0)
        if art_count == 0 or (oldest_hours is not None and oldest_hours > 72):
            triggers.append("STALE_NEWS")

        sent_score = news_data.get("sentiment_score", 0.0)
        if abs(sent_score) < 0.05:
            triggers.append("NEUTRAL_SENT")

        if not triggers:
            return {
                "needs_more_research": False,
                "triggers_fired":      [],
                "reasoning":           "All quality checks passed – proceeding to synthesis.",
                "recommended_actions": [],
            }

        # ── LLM pass only when at least one trigger fired ─────────────────────
        payload = {
            "query":   query,
            "results": results,
            "rule_based_triggers": triggers,
        }
        resp = await self._client.chat.completions.create(
            model=_settings.llm_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": json.dumps(payload)},
            ],
        )
        return json.loads(resp.choices[0].message.content)
