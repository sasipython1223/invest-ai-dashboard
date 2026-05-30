from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.data.asset_classifier import is_tradeable_asset
from src.data.ticker_mapper import get_data_ticker

try:
    import yfinance as yf
except Exception:  # pragma: no cover - defensive import guard
    yf = None


def _mock_price_history(
    ticker: str,
    periods: int = 260,
    end_date: datetime | None = None,
) -> pd.Series:
    """Deterministic mock close-price history for safe local prototyping."""
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    dates = pd.date_range(end=end_date.date(), periods=periods, freq="B")
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


def load_prices_for_watchlist(
    watchlist: pd.DataFrame | list[str],
) -> dict[str, pd.Series]:
    """Load history for each ticker, skipping non-tradeable reserve assets."""
    if isinstance(watchlist, list):
        return {ticker: load_price_history(ticker=ticker) for ticker in watchlist}

    prices: dict[str, pd.Series] = {}
    for _, row in watchlist.iterrows():
        ticker = str(row["ticker"])
        if is_tradeable_asset(row):
            data_ticker = get_data_ticker(row) or ticker
            prices[ticker] = load_price_history(ticker=data_ticker)
        else:
            prices[ticker] = pd.Series(dtype=float, name="Close")
    return prices
