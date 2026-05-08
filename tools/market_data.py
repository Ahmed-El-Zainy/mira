"""tools/market_data.py – fetches live market data via yfinance."""
from __future__ import annotations

import asyncio
from typing import Any

import yfinance as yf


class MarketDataTool:
    name = "market_data"

    async def execute(self, ticker: str) -> dict[str, Any]:
        """Fetch structured market data for *ticker*. Raises ValueError on unknown ticker."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._fetch, ticker)

    def _fetch(self, ticker: str) -> dict[str, Any]:
        stock = yf.Ticker(ticker)
        info = stock.info

        # yfinance returns a minimal dict (e.g. only {"trailingPegRatio": None})
        # for delisted / unknown tickers – treat as not found.
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            # Try history as a fallback indicator
            hist = stock.history(period="5d")
            if hist.empty:
                raise ValueError(f"Ticker '{ticker}' not found or has no tradable data.")

        history_1y = stock.history(period="1y")
        high_52w = float(history_1y["High"].max()) if not history_1y.empty else None
        low_52w  = float(history_1y["Low"].min())  if not history_1y.empty else None

        quarterly_revenues: list[float] = []
        try:
            fin = stock.quarterly_financials
            if "Total Revenue" in fin.index:
                quarterly_revenues = [
                    float(v) for v in fin.loc["Total Revenue"].iloc[:2].tolist()
                    if v is not None
                ]
        except Exception:
            pass

        return {
            "ticker": ticker.upper(),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "daily_change_pct": info.get("regularMarketChangePercent"),
            "volume": info.get("volume"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "52w_high": high_52w,
            "52w_low":  low_52w,
            "quarterly_revenues": quarterly_revenues,
            "company_name": info.get("longName") or info.get("shortName", ticker.upper()),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
