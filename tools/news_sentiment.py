"""tools/news_sentiment.py – fetches news via NewsAPI and scores via FinBERT.

Optimisations vs. the original implementation:
  * News fetch uses async httpx (not blocking `requests`) so we don't stall the
    event loop while waiting on NewsAPI.
  * FinBERT is invoked in a single batched call instead of N sequential calls
    (transformers pipeline supports list inputs natively, ~3-5x faster).
  * NewsAPI responses are cached for 5 minutes per (company, ticker) tuple to
    avoid repeated calls when the same ticker is analysed in quick succession.

Trade-offs documented in README:
  * FinBERT (local)  – no API cost, domain-tuned for finance, ~200 ms/article,
    but requires ~500 MB model download on first run and GPU is optional.
  * LLM-based        – zero extra setup, GPT-4 is highly context-aware, but
    adds latency & cost per article. FinBERT is the better default here.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import httpx

logger = logging.getLogger("mira.news_sentiment")

_FINBERT_MODEL = "ProsusAI/finbert"
_NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Simple in-process TTL cache for the NewsAPI response.
_NEWS_CACHE: dict[str, tuple[float, list[dict]]] = {}
_NEWS_TTL_SECONDS = 300  # 5 min


@lru_cache(maxsize=1)
def _get_pipeline():
    """Lazy-load FinBERT so it doesn't block startup."""
    try:
        from transformers import pipeline  # type: ignore
        return pipeline("sentiment-analysis", model=_FINBERT_MODEL, truncation=True)
    except (AttributeError, ImportError) as e:
        raise RuntimeError(f"FinBERT initialization failed: {str(e)}")


async def preload_finbert() -> None:
    """Warm up FinBERT in a background thread so the first /analyze isn't
    blocked on the ~500MB model download. Safe to call multiple times — the
    `lru_cache` on `_get_pipeline` makes subsequent calls free.

    Errors are swallowed (just logged) — sentiment analysis will fall back to
    neutral scores if the download ultimately fails.
    """
    def _warm() -> None:
        try:
            t0 = time.time()
            _get_pipeline()
            logger.info("FinBERT preloaded in %.1fs.", time.time() - t0)
        except Exception as exc:
            logger.warning("FinBERT preload failed: %s", exc)

    await asyncio.to_thread(_warm)


def _article_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


class NewsSentimentTool:
    name = "news_sentiment"

    def __init__(self, news_api_key: str) -> None:
        self._key = news_api_key

    async def execute(self, company_name: str, ticker: str) -> dict[str, Any]:
        articles = await self._fetch_articles(company_name, ticker)
        return await asyncio.to_thread(self._score, articles)

    # ── scoring (CPU-bound, runs in a worker thread) ─────────────────────────
    def _score(self, articles: list[dict]) -> dict[str, Any]:
        if not articles:
            return {
                "articles": [],
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0},
                "sentiment_score": 0.0,
                "article_count": 0,
                "oldest_article_hours": None,
            }

        try:
            nlp = _get_pipeline()
        except Exception as e:
            logger.warning("Local FinBERT failed: %s. Falling back to neutral scores.", e)
            return {
                "articles": articles,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": len(articles)},
                "sentiment_score": 0.0,
                "article_count": len(articles),
                "error": str(e),
            }

        # Batch all texts into a single pipeline call.
        texts: list[str] = [
            f"{a.get('title', '')}. {a.get('description') or ''}"[:512] for a in articles
        ]
        scored_batch = nlp(texts) if texts else []

        results: list[dict] = []
        dist: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        oldest_hours: float | None = None

        for art, scored in zip(articles, scored_batch):
            label: str = scored["label"].lower()
            dist[label] = dist.get(label, 0) + 1

            published = art.get("publishedAt")
            age_h: float | None = None
            if published:
                try:
                    pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                    age_h = (datetime.now(timezone.utc) - pub_dt).total_seconds() / 3600
                    if oldest_hours is None or age_h > oldest_hours:
                        oldest_hours = age_h
                except Exception:
                    pass

            results.append({
                "id": _article_id(art.get("url", "")),
                "title": art.get("title"),
                "url": art.get("url"),
                "published_at": published,
                "age_hours": round(age_h, 1) if age_h is not None else None,
                "sentiment": label,
                "confidence": round(float(scored["score"]), 4),
            })

        total = len(results)
        score = (dist["positive"] - dist["negative"]) / total if total else 0.0

        return {
            "articles": results,
            "sentiment_distribution": dist,
            "sentiment_score": round(score, 4),
            "article_count": total,
            "oldest_article_hours": round(oldest_hours, 1) if oldest_hours else None,
        }

    # ── async fetch with TTL cache ───────────────────────────────────────────
    async def _fetch_articles(self, company_name: str, ticker: str) -> list[dict]:
        if not self._key:
            return []

        cache_key = f"{company_name}|{ticker}".lower()
        now = time.time()
        cached = _NEWS_CACHE.get(cache_key)
        if cached and (now - cached[0]) < _NEWS_TTL_SECONDS:
            return cached[1]

        params = {
            "q": f'"{company_name}" OR "{ticker}"',
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": "5",
        }
        headers = {"Authorization": f"Bearer {self._key}"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(_NEWSAPI_URL, params=params, headers=headers)
                resp.raise_for_status()
                articles = resp.json().get("articles", []) or []
        except Exception as exc:
            logger.warning("NewsAPI fetch failed: %s", exc)
            return []

        _NEWS_CACHE[cache_key] = (now, articles)
        return articles
