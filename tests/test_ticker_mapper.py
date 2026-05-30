import pandas as pd
import pytest

from src.data.ticker_mapper import get_data_ticker, has_valid_data_ticker


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"ticker": "DBS", "data_ticker": "D05.SI"}, "D05.SI"),
        ({"ticker": "UOB", "data_ticker": "U11.SI"}, "U11.SI"),
        ({"ticker": "ES3", "data_ticker": "ES3.SI"}, "ES3.SI"),
        ({"ticker": "VWRA", "data_ticker": "VWRA.L"}, "VWRA.L"),
        ({"ticker": "CSPX", "data_ticker": "CSPX.L"}, "CSPX.L"),
        ({"ticker": "AAPL", "data_ticker": "AAPL"}, "AAPL"),
        # Blank data_ticker -> falls back to display ticker
        ({"ticker": "CASH", "data_ticker": ""}, "CASH"),
        # Missing data_ticker key -> falls back to display ticker
        ({"ticker": "MSFT"}, "MSFT"),
    ],
)
def test_get_data_ticker(row, expected):
    assert get_data_ticker(row) == expected


def test_get_data_ticker_with_series():
    row = pd.Series({"ticker": "DBS", "data_ticker": "D05.SI"})
    assert get_data_ticker(row) == "D05.SI"


def test_has_valid_data_ticker_true():
    assert has_valid_data_ticker({"ticker": "AAPL", "data_ticker": "AAPL"}) is True


def test_has_valid_data_ticker_false_when_both_empty():
    assert has_valid_data_ticker({"ticker": "", "data_ticker": ""}) is False
