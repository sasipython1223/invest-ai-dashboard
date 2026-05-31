import inspect
import socket

import numpy as np
import pandas as pd

from src.analytics import scenario_forecast
from src.analytics.scenario_forecast import (
    DEFAULT_HORIZONS,
    assign_volatility_risk_status,
    build_scenario_summary,
    build_volatility_cone,
    calculate_annualized_return_volatility,
    calculate_log_returns,
)


def _build_price_series(length: int = 200, start: float = 100.0, drift: float = 0.001) -> pd.Series:
    values = [start]
    for _ in range(length - 1):
        values.append(values[-1] * (1.0 + drift))
    return pd.Series(values, dtype=float)


def test_calculate_log_returns_uses_log_ratio():
    prices = pd.Series([100.0, 110.0, 121.0], dtype=float)

    log_returns = calculate_log_returns(prices)

    expected = pd.Series([np.log(110.0 / 100.0), np.log(121.0 / 110.0)])
    assert np.allclose(log_returns.values, expected.values)


def test_calculate_annualized_return_and_volatility_matches_formula():
    log_returns = pd.Series([0.01, 0.02, -0.005, 0.015], dtype=float)

    stats = calculate_annualized_return_volatility(log_returns)

    assert stats["annual_return"] == log_returns.mean() * 252
    assert stats["annual_volatility"] == log_returns.std() * np.sqrt(252)


def test_insufficient_data_returns_empty_outputs():
    short_prices = _build_price_series(length=40)

    assert build_volatility_cone(short_prices).empty
    assert build_scenario_summary(short_prices).empty


def test_volatility_cone_starts_at_current_value():
    prices = _build_price_series(length=200)
    current_value = float(prices.iloc[-1])

    cone_df = build_volatility_cone(prices, horizon_days=126)

    assert not cone_df.empty
    first_row = cone_df.iloc[0]
    assert first_row["step"] == 0
    assert first_row["expected"] == current_value
    assert first_row["normal_upper"] == current_value
    assert first_row["normal_lower"] == current_value
    assert first_row["extreme_upper"] == current_value
    assert first_row["extreme_lower"] == current_value


def test_expected_path_is_between_normal_bands():
    prices = _build_price_series(length=220, drift=0.002)

    cone_df = build_volatility_cone(prices, horizon_days=126)

    assert not cone_df.empty
    assert (cone_df["expected"] <= cone_df["normal_upper"]).all()
    assert (cone_df["expected"] >= cone_df["normal_lower"]).all()


def test_normal_band_is_inside_extreme_band():
    prices = _build_price_series(length=220, drift=0.0015)

    cone_df = build_volatility_cone(prices, horizon_days=126)

    assert not cone_df.empty
    assert (cone_df["normal_upper"] <= cone_df["extreme_upper"]).all()
    assert (cone_df["normal_lower"] >= cone_df["extreme_lower"]).all()


def test_scenario_summary_produces_1m_3m_6m_rows():
    prices = _build_price_series(length=230, drift=0.001)

    summary_df = build_scenario_summary(prices)

    assert not summary_df.empty
    assert summary_df["horizon"].tolist() == list(DEFAULT_HORIZONS.keys())


def test_assign_volatility_risk_status_thresholds():
    assert assign_volatility_risk_status(0.10) == "Low"
    assert assign_volatility_risk_status(0.15) == "Moderate"
    assert assign_volatility_risk_status(0.30) == "Moderate"
    assert assign_volatility_risk_status(0.31) == "Elevated"
    assert assign_volatility_risk_status(0.50) == "Elevated"
    assert assign_volatility_risk_status(0.51) == "High"


def test_scenario_forecast_module_has_no_external_api_or_ai_calls(monkeypatch):
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network call attempted")),
    )

    prices = _build_price_series(length=200, drift=0.0012)
    cone_df = build_volatility_cone(prices)
    summary_df = build_scenario_summary(prices)

    assert not cone_df.empty
    assert not summary_df.empty
    module_source = inspect.getsource(scenario_forecast)
    assert "yfinance" not in module_source
    assert "review_with_gemini" not in module_source
    assert "review_with_openai" not in module_source
