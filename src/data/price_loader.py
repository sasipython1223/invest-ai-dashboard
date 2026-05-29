from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - defensive import guard
    yf = None


def _mock_price_history(ticker: str, periods: int = 260) -> pd.Series:
    """Deterministic mock close-price history for safe local prototyping."""
    end = datetime.utcnow().date()
    dates = pd.date_range(end=end, periods=periods, freq="B")
    base = max(20, sum(ord(char) for char in ticker) % 200)
    values = np.linspace(base * 0.85, base * 1.15, periods)
    return pd.Series(values, index=dates, name="Close")


def load_price_history(ticker: str, period: str = "18mo") -> pd.Series:
    """Load close prices from yfinance with deterministic mock fallback."""
    if yf is None:
        return _mock_price_history(ticker)

    try:
        history = yf.Ticker(ticker).history(period=period)
        if history.empty or "Close" not in history:
            return _mock_price_history(ticker)
        return history["Close"].dropna()
    except Exception:
        return _mock_price_history(ticker)


def load_prices_for_watchlist(tickers: list[str]) -> dict[str, pd.Series]:
    """Load history for each ticker, always returning safe data."""
    return {ticker: load_price_history(ticker=ticker) for ticker in tickers}
