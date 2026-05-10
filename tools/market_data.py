"""tools/market_data.py – fetches live market data via yfinance.

Optimisations vs. the original implementation:
  * Uses `asyncio.to_thread` instead of the deprecated `get_event_loop().run_in_executor`.
  * Short-lived in-process cache (60 s) so successive analyses of the same
    ticker do not hammer Yahoo three times.
  * Wraps the yfinance call in a single try/except — yfinance occasionally
    raises generic `Exception` from `info` for delisted tickers.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import yfinance as yf

logger = logging.getLogger("mira.market_data")

_CACHE_TTL_SECONDS = 60
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


class MarketDataTool:
    name = "market_data"

    async def execute(self, ticker: str) -> dict[str, Any]:
        """Fetch structured market data for *ticker*. Raises ValueError on unknown ticker."""
        ticker_u = ticker.upper()
        now = time.time()
        cached = _CACHE.get(ticker_u)
        if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]

        data = await asyncio.to_thread(self._fetch, ticker_u)
        _CACHE[ticker_u] = (now, data)
        return data

    def _fetch(self, ticker: str) -> dict[str, Any]:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
        except Exception as exc:
            raise ValueError(f"Ticker '{ticker}' not found or unavailable: {exc}") from exc

        if not info or (
            info.get("regularMarketPrice") is None and info.get("currentPrice") is None
        ):
            hist = stock.history(period="5d")
            if hist.empty:
                raise ValueError(f"Ticker '{ticker}' not found or has no tradable data.")

        history_1y = stock.history(period="1y")
        high_52w = float(history_1y["High"].max()) if not history_1y.empty else None
        low_52w = float(history_1y["Low"].min()) if not history_1y.empty else None

        quarterly_revenues: list[float] = []
        try:
            fin = stock.quarterly_financials
            if "Total Revenue" in fin.index:
                quarterly_revenues = [
                    float(v) for v in fin.loc["Total Revenue"].iloc[:2].tolist()
                    if v is not None
                ]
        except Exception as exc:
            logger.debug("Quarterly financials unavailable for %s: %s", ticker, exc)

        return {
            "ticker": ticker.upper(),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "daily_change_pct": info.get("regularMarketChangePercent"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52w_high": high_52w,
            "52w_low": low_52w,
            "quarterly_revenues": quarterly_revenues,
            "company_name": info.get("longName") or info.get("shortName", ticker.upper()),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
