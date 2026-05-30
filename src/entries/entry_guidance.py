from __future__ import annotations

import pandas as pd

from src.data.asset_classifier import is_tradeable_asset

STATUS_OK = "OK"
STATUS_INSUFFICIENT = "Insufficient Data"
STATUS_NOT_APPLICABLE = "Not Applicable"


def _to_float(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _sma(price_history: pd.Series, window: int) -> float | None:
    if len(price_history) < window:
        return None
    return _to_float(price_history.tail(window).mean())


def _atr_like(price_history: pd.Series, window: int = 14) -> float | None:
    if len(price_history) < 2:
        return None
    true_ranges = price_history.diff().abs().dropna()
    if true_ranges.empty:
        return None
    span = true_ranges.tail(window)
    return _to_float(span.mean())


def _format_zone(low: float | None, high: float | None) -> str | None:
    if low is None or high is None:
        return None
    zone_low = min(low, high)
    zone_high = max(low, high)
    return f"{zone_low:.3f} to {zone_high:.3f}"


def _base_payload(ticker: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "status": STATUS_OK,
        "latest_price": None,
        "recent_close": None,
        "sma_20": None,
        "sma_50": None,
        "sma_200": None,
        "atr_14": None,
        "recent_low_20d": None,
        "recent_high_20d": None,
        "aggressive_bid_zone": None,
        "normal_bid_zone": None,
        "conservative_bid_zone": None,
        "aggressive_bid_zone_bounds": {"low": None, "high": None},
        "normal_bid_zone_bounds": {"low": None, "high": None},
        "conservative_bid_zone_bounds": {"low": None, "high": None},
        "preferred_order_type": "Limit order only",
        "manual_review_required": True,
        "warnings": [
            "Manual review required. Final execution must remain outside this dashboard.",
            "No auto-trading, broker API, or Tiger API integration is provided.",
            "Do not use market orders by default; review bid/ask spread and place manual limit orders only.",
            "No price zone is guaranteed safe. Re-check live prices before any manual order.",
        ],
    }


def calculate_entry_guidance(
    ticker: str,
    price_history: pd.Series,
    signal_row: pd.Series | dict,
) -> dict[str, object]:
    payload = _base_payload(ticker)

    if not is_tradeable_asset(signal_row):
        payload["status"] = STATUS_NOT_APPLICABLE
        payload["warnings"] = [
            f"{ticker} is marked as a non-tradeable reserve asset. Entry guidance is not applicable.",
            *payload["warnings"],
        ]
        return payload

    history = pd.Series(price_history, dtype=float).dropna()
    count = len(history)
    if count < 200:
        payload["status"] = STATUS_INSUFFICIENT
        payload["latest_price"] = _to_float(history.iloc[-1]) if count else None
        payload["recent_close"] = payload["latest_price"]
        payload["warnings"] = [
            f"Need at least 200 data points for full ETF entry guidance; received {count}.",
            *payload["warnings"],
        ]
        return payload

    latest_price = _to_float(history.iloc[-1])
    sma_20 = _sma(history, 20)
    sma_50 = _sma(history, 50)
    sma_200 = _sma(history, 200)
    atr_14 = _atr_like(history, 14) or 0.0
    recent_low_20d = _to_float(history.tail(20).min())
    recent_high_20d = _to_float(history.tail(20).max())

    aggressive_low = max(sma_20 if sma_20 is not None else latest_price, latest_price - 0.5 * atr_14)
    aggressive_high = latest_price

    normal_upper = min(latest_price, max(sma_20 if sma_20 is not None else latest_price, sma_50 if sma_50 is not None else latest_price))
    normal_lower = min(normal_upper, latest_price - atr_14)

    conservative_upper = min(latest_price, sma_50 if sma_50 is not None else latest_price)
    conservative_lower = min(conservative_upper, latest_price - 1.5 * atr_14)

    payload.update(
        {
            "latest_price": latest_price,
            "recent_close": latest_price,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "atr_14": _to_float(atr_14),
            "recent_low_20d": recent_low_20d,
            "recent_high_20d": recent_high_20d,
            "aggressive_bid_zone": _format_zone(aggressive_low, aggressive_high),
            "normal_bid_zone": _format_zone(normal_lower, normal_upper),
            "conservative_bid_zone": _format_zone(conservative_lower, conservative_upper),
            "aggressive_bid_zone_bounds": {
                "low": _to_float(aggressive_low),
                "high": _to_float(aggressive_high),
            },
            "normal_bid_zone_bounds": {
                "low": _to_float(normal_lower),
                "high": _to_float(normal_upper),
            },
            "conservative_bid_zone_bounds": {
                "low": _to_float(conservative_lower),
                "high": _to_float(conservative_upper),
            },
        }
    )

    return payload
