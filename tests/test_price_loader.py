import pandas as pd

from src.data import price_loader


def test_load_prices_for_watchlist_skips_cash_history(monkeypatch):
    watchlist = pd.DataFrame(
        [
            {"ticker": "AAPL", "market": "NASDAQ", "type": "Stock"},
            {"ticker": "CASH", "market": "CASH", "type": "Cash"},
        ]
    )
    calls: list[str] = []

    def fake_load_price_history(ticker: str, period: str = "18mo") -> pd.Series:
        calls.append(ticker)
        return pd.Series([1.0, 2.0], name="Close")

    monkeypatch.setattr(price_loader, "load_price_history", fake_load_price_history)

    prices = price_loader.load_prices_for_watchlist(watchlist)

    assert calls == ["AAPL"]
    assert list(prices["AAPL"]) == [1.0, 2.0]
    assert prices["CASH"].empty


def test_load_prices_uses_data_ticker_for_yfinance(monkeypatch):
    """price_loader must call load_price_history with data_ticker, not display ticker."""
    watchlist = pd.DataFrame(
        [
            {"ticker": "DBS", "data_ticker": "D05.SI", "market": "SGX", "type": "Stock"},
            {"ticker": "UOB", "data_ticker": "U11.SI", "market": "SGX", "type": "Stock"},
            {"ticker": "CASH", "data_ticker": "CASH", "market": "CASH", "type": "Cash"},
        ]
    )
    calls: list[str] = []

    def fake_load_price_history(ticker: str, period: str = "18mo") -> pd.Series:
        calls.append(ticker)
        return pd.Series([10.0], name="Close")

    monkeypatch.setattr(price_loader, "load_price_history", fake_load_price_history)

    prices = price_loader.load_prices_for_watchlist(watchlist)

    # yfinance calls use provider symbols
    assert "D05.SI" in calls
    assert "U11.SI" in calls
    # CASH must not be fetched
    assert "CASH" not in calls

    # Result dict is keyed by display ticker
    assert "DBS" in prices
    assert "UOB" in prices
    assert prices["CASH"].empty

