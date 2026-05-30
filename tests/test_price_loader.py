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
