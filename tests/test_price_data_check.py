import pandas as pd

from src.utils.price_data_check import has_price_data


def test_returns_false_for_non_dict():
    assert has_price_data(None) is False
    assert has_price_data([]) is False
    assert has_price_data(pd.DataFrame()) is False


def test_returns_false_for_empty_dict():
    assert has_price_data({}) is False


def test_returns_false_when_all_values_empty():
    assert has_price_data({"AAPL": pd.DataFrame()}) is False


def test_returns_true_for_non_empty_dataframe_value():
    df = pd.DataFrame({"close": [100.0, 101.0]})
    assert has_price_data({"AAPL": df}) is True


def test_returns_true_for_non_empty_list_value():
    assert has_price_data({"AAPL": [1, 2, 3]}) is True


def test_returns_false_for_empty_list_value():
    assert has_price_data({"AAPL": []}) is False


def test_returns_true_when_at_least_one_value_non_empty():
    assert has_price_data({"A": pd.DataFrame(), "B": pd.DataFrame({"c": [1]})}) is True
