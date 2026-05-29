import pandas as pd
import pytest

from src.data.watchlist_loader import load_watchlist


def test_load_watchlist_success(tmp_path):
    path = tmp_path / "watchlist.csv"
    path.write_text(
        "ticker,name,market,type,bucket,currency,target_weight\n"
        "AAPL,Apple,NASDAQ,Stock,tactical,USD,3\n",
        encoding="utf-8",
    )

    df = load_watchlist(path)

    assert list(df.columns) == [
        "ticker",
        "name",
        "market",
        "type",
        "bucket",
        "currency",
        "target_weight",
        "is_tradeable",
    ]
    assert df.loc[0, "ticker"] == "AAPL"
    assert df.loc[0, "target_weight"] == 3
    assert bool(df.loc[0, "is_tradeable"])


def test_load_watchlist_marks_cash_as_non_tradeable(tmp_path):
    path = tmp_path / "watchlist.csv"
    path.write_text(
        "ticker,name,market,type,bucket,currency,target_weight\n"
        "CASH,Cash Reserve,CASH,Cash,cash,SGD,10\n",
        encoding="utf-8",
    )

    df = load_watchlist(path)

    assert not bool(df.loc[0, "is_tradeable"])


def test_load_watchlist_missing_columns(tmp_path):
    path = tmp_path / "watchlist.csv"
    pd.DataFrame([{"ticker": "AAPL"}]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing watchlist columns"):
        load_watchlist(path)
