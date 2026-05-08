"""tools/peer_correlation.py – real Pearson correlation vs. indices, sector ETFs, peers."""
from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import yfinance as yf

# Sector ETF mapping (GICS-aligned)
_SECTOR_ETF: dict[str, str] = {
    "Technology":              "XLK",
    "Communication Services":  "XLC",
    "Consumer Cyclical":       "XLY",
    "Consumer Defensive":      "XLP",
    "Energy":                  "XLE",
    "Financial Services":      "XLF",
    "Healthcare":              "XLV",
    "Industrials":             "XLI",
    "Basic Materials":         "XLB",
    "Real Estate":             "XLRE",
    "Utilities":               "XLU",
}

# Default peer tickers per sector
_DEFAULT_PEERS: dict[str, list[str]] = {
    "Technology":             ["MSFT", "GOOGL"],
    "Communication Services": ["META", "NFLX"],
    "Consumer Cyclical":      ["AMZN", "NIO"],
    "Consumer Defensive":     ["PG", "KO"],
    "Energy":                 ["XOM", "CVX"],
    "Financial Services":     ["JPM", "BAC"],
    "Healthcare":             ["JNJ", "PFE"],
    "Industrials":            ["HON", "GE"],
    "Basic Materials":        ["LIN", "APD"],
    "Real Estate":            ["PLD", "AMT"],
    "Utilities":              ["NEE", "DUK"],
}


class PeerCorrelationTool:
    name = "peer_correlation"

    async def execute(
        self,
        ticker: str,
        sector: str | None = None,
        peers: list[str] | None = None,
    ) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._compute, ticker, sector, peers)

    def _compute(
        self,
        ticker: str,
        sector: str | None,
        peers: list[str] | None,
    ) -> dict[str, Any]:
        sector_etf = _SECTOR_ETF.get(sector or "", "SPY")
        if peers is None:
            peers = _DEFAULT_PEERS.get(sector or "", ["SPY"])[:2]

        all_symbols = list({ticker.upper(), "^GSPC", sector_etf, *peers})

        try:
            raw = yf.download(all_symbols, period="1y", auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]] if "Close" in raw.columns else raw
        except Exception as exc:
            return {"error": str(exc)}

        returns = prices.pct_change().dropna()
        t = ticker.upper()
        if t not in returns.columns:
            return {"error": f"No return data for {ticker}"}

        correlations: dict[str, float | None] = {}
        for sym in all_symbols:
            if sym != t and sym in returns.columns:
                correlations[sym] = round(float(returns[t].corr(returns[sym])), 4)

        return {
            "market_correlation":  correlations.get("^GSPC"),
            "sector_etf":          sector_etf,
            "sector_correlation":  correlations.get(sector_etf),
            "peer_correlations":   {p: correlations.get(p) for p in peers},
            "all_correlations":    {**{t: 1.0}, **correlations},
        }
