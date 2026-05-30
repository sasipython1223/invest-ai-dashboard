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

    assert "ticker" in df.columns
    assert "is_tradeable" in df.columns
    assert "data_ticker" in df.columns
    assert df.loc[0, "ticker"] == "AAPL"
    assert df.loc[0, "target_weight"] == 3
    assert bool(df.loc[0, "is_tradeable"])


def test_load_watchlist_data_ticker_defaults_to_ticker_when_missing(tmp_path):
    """When CSV has no data_ticker column, data_ticker defaults to display ticker."""
    path = tmp_path / "watchlist.csv"
    path.write_text(
        "ticker,name,market,type,bucket,currency,target_weight\n"
        "AAPL,Apple,NASDAQ,Stock,tactical,USD,3\n",
        encoding="utf-8",
    )

    df = load_watchlist(path)

    assert df.loc[0, "data_ticker"] == "AAPL"


def test_load_watchlist_data_ticker_uses_explicit_value(tmp_path):
    """When CSV includes data_ticker, it is used as-is (after strip)."""
    path = tmp_path / "watchlist.csv"
    path.write_text(
        "ticker,data_ticker,name,market,type,bucket,currency,target_weight\n"
        "DBS,D05.SI,DBS Group,SGX,Stock,singapore,SGD,5\n"
        "UOB,U11.SI,UOB,SGX,Stock,singapore,SGD,5\n"
        "ES3,ES3.SI,STI ETF,SGX,ETF,singapore,SGD,10\n"
        "VWRA,VWRA.L,Vanguard FTSE All-World,LSE,ETF,core,USD,40\n"
        "CASH,,Cash Reserve,CASH,Cash,cash,SGD,12\n",
        encoding="utf-8",
    )

    df = load_watchlist(path).set_index("ticker")

    assert df.loc["DBS", "data_ticker"] == "D05.SI"
    assert df.loc["UOB", "data_ticker"] == "U11.SI"
    assert df.loc["ES3", "data_ticker"] == "ES3.SI"
    assert df.loc["VWRA", "data_ticker"] == "VWRA.L"
    # CASH has blank data_ticker -> falls back to display ticker
    assert df.loc["CASH", "data_ticker"] == "CASH"


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
