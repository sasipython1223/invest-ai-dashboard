import numpy as np
import pandas as pd

from src.entries.entry_guidance import (
    STATUS_INSUFFICIENT,
    STATUS_NOT_APPLICABLE,
    STATUS_OK,
    calculate_entry_guidance,
)
from src.ui.charts import build_entry_zone_dataframe


def _series(values):
    return pd.Series(np.array(values, dtype=float))


def _history_with_atr(length: int = 240) -> pd.Series:
    base = np.linspace(4.6, 5.2, length)
    noise = 0.02 * np.sin(np.arange(length))
    return _series(base + noise)


def test_entry_guidance_with_sufficient_history():
    price_history = _history_with_atr()

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
    price_history = _history_with_atr(220)

    result = calculate_entry_guidance(
        ticker="ES3",
        price_history=price_history,
        signal_row={"type": "ETF", "market": "SGX"},
    )

    assert result["conservative_bid_zone_bounds"]["high"] <= result["latest_price"]


def test_normal_and_conservative_bid_zones_are_practical_ranges_when_atr_available():
    result = calculate_entry_guidance(
        ticker="ES3",
        price_history=_history_with_atr(220),
        signal_row={"type": "ETF", "market": "SGX"},
    )

    normal_bounds = result["normal_bid_zone_bounds"]
    conservative_bounds = result["conservative_bid_zone_bounds"]

    assert result["atr_14"] > 0
    assert normal_bounds["high"] > normal_bounds["low"]
    assert conservative_bounds["high"] > conservative_bounds["low"]


def test_all_bid_zone_bounds_are_normalized_and_capped_at_latest_price():
    result = calculate_entry_guidance(
        ticker="ES3",
        price_history=_history_with_atr(230),
        signal_row={"type": "ETF", "market": "SGX"},
    )

    for key in [
        "aggressive_bid_zone_bounds",
        "normal_bid_zone_bounds",
        "conservative_bid_zone_bounds",
    ]:
        bounds = result[key]
        assert bounds["low"] <= bounds["high"]
        assert bounds["high"] <= result["latest_price"]


def test_atr_fallback_creates_non_zero_practical_ranges_when_atr_is_zero():
    flat_history = _series(np.full(220, 5.0))
    result = calculate_entry_guidance(
        ticker="ES3",
        price_history=flat_history,
        signal_row={"type": "ETF", "market": "SGX"},
    )

    assert result["atr_14"] == 0.0
    assert result["normal_bid_zone_bounds"]["high"] > result["normal_bid_zone_bounds"]["low"]
    assert result["conservative_bid_zone_bounds"]["high"] > result["conservative_bid_zone_bounds"]["low"]
    assert any("Bid zone is narrow" in warning for warning in result["warnings"])


def test_entry_zone_visual_helper_accepts_updated_bounds_shape():
    result = calculate_entry_guidance(
        ticker="ES3",
        price_history=_history_with_atr(230),
        signal_row={"type": "ETF", "market": "SGX"},
    )

    entry_zone_df = build_entry_zone_dataframe(result)

    assert len(entry_zone_df) == 3
    assert set(entry_zone_df["label"]) == {
        "Aggressive Bid Zone",
        "Normal Bid Zone",
        "Conservative Bid Zone",
    }


def test_cash_asset_returns_not_applicable():
    price_history = _series(np.linspace(1.0, 1.0, 220))

    result = calculate_entry_guidance(
        ticker="CASH",
        price_history=price_history,
        signal_row={"type": "Cash", "market": "CASH"},
    )

    assert result["status"] == STATUS_NOT_APPLICABLE
    assert "not applicable" in result["warnings"][0].lower()
