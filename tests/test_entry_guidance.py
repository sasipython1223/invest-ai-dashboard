import numpy as np
import pandas as pd

from src.entries.entry_guidance import (
    STATUS_INSUFFICIENT,
    STATUS_NOT_APPLICABLE,
    STATUS_OK,
    calculate_entry_guidance,
)


def _series(values):
    return pd.Series(np.array(values, dtype=float))


def test_entry_guidance_with_sufficient_history():
    price_history = _series(np.linspace(4.6, 5.2, 240))

    result = calculate_entry_guidance(
        ticker="ES3",
        price_history=price_history,
        signal_row={"type": "ETF", "market": "SGX", "signal": "Hold / Buy Candidate"},
    )

    assert result["status"] == STATUS_OK
    assert result["latest_price"] is not None
    assert result["sma_20"] is not None
    assert result["sma_50"] is not None
    assert result["sma_200"] is not None
    assert result["atr_14"] is not None
    assert result["preferred_order_type"] == "Limit order only"
    assert result["manual_review_required"] is True


def test_entry_guidance_insufficient_data_fallback():
    price_history = _series(np.linspace(5.0, 5.1, 60))

    result = calculate_entry_guidance(
        ticker="ES3",
        price_history=price_history,
        signal_row={"type": "ETF", "market": "SGX"},
    )

    assert result["status"] == STATUS_INSUFFICIENT
    assert "Need at least 200 data points" in result["warnings"][0]


def test_conservative_bid_zone_not_above_latest_price():
    price_history = _series(np.linspace(4.7, 5.3, 220))

    result = calculate_entry_guidance(
        ticker="ES3",
        price_history=price_history,
        signal_row={"type": "ETF", "market": "SGX"},
    )

    assert result["conservative_bid_zone_bounds"]["high"] <= result["latest_price"]


def test_cash_asset_returns_not_applicable():
    price_history = _series(np.linspace(1.0, 1.0, 220))

    result = calculate_entry_guidance(
        ticker="CASH",
        price_history=price_history,
        signal_row={"type": "Cash", "market": "CASH"},
    )

    assert result["status"] == STATUS_NOT_APPLICABLE
    assert "not applicable" in result["warnings"][0].lower()
