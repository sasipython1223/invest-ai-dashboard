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


def _normalize_bounds(low: float, high: float, *, upper_cap: float | None = None) -> tuple[float, float]:
    zone_low = min(low, high)
    zone_high = max(low, high)
    if upper_cap is not None:
        zone_high = min(zone_high, upper_cap)
        zone_low = min(zone_low, zone_high)
    return zone_low, zone_high


def _build_buffered_bid_zone(center: float, buffer: float, latest_price: float) -> tuple[float, float]:
    zone_low, zone_high = _normalize_bounds(center - buffer, center + buffer, upper_cap=latest_price)
    if buffer > 0 and zone_high - zone_low <= 0:
        zone_low = max(0.0, zone_high - buffer)
    return _normalize_bounds(zone_low, zone_high, upper_cap=latest_price)


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
    raw_atr_14 = _atr_like(history, 14)
    has_valid_atr = raw_atr_14 is not None and raw_atr_14 > 0
    fallback_buffer = latest_price * 0.0025
    atr_14 = raw_atr_14 if has_valid_atr else 0.0
    range_buffer = max(raw_atr_14 * 0.25, fallback_buffer) if has_valid_atr else fallback_buffer
    atr_for_centers = raw_atr_14 if has_valid_atr else range_buffer * 4
    recent_low_20d = _to_float(history.tail(20).min())
    recent_high_20d = _to_float(history.tail(20).max())

    aggressive_anchor = latest_price - 0.5 * atr_for_centers
    aggressive_floor = sma_20 if sma_20 is not None else aggressive_anchor
    aggressive_low, aggressive_high = _normalize_bounds(
        max(aggressive_floor, aggressive_anchor),
        latest_price,
        upper_cap=latest_price,
    )

    normal_center = sma_20 if sma_20 is not None else latest_price - atr_for_centers
    normal_lower, normal_upper = _build_buffered_bid_zone(normal_center, range_buffer, latest_price)

    conservative_center = sma_50 if sma_50 is not None else latest_price - 1.5 * atr_for_centers
    conservative_lower, conservative_upper = _build_buffered_bid_zone(
        conservative_center,
        range_buffer,
        latest_price,
    )

    narrow_threshold = max(latest_price * 0.001, range_buffer * 0.5)
    zone_widths = [
        aggressive_high - aggressive_low,
        normal_upper - normal_lower,
        conservative_upper - conservative_lower,
    ]
    narrow_zone_warning = (
        "Bid zone is narrow; verify live bid/ask spread before placing a manual limit order."
    )
    if any(width <= narrow_threshold for width in zone_widths) and narrow_zone_warning not in payload["warnings"]:
        payload["warnings"] = [*payload["warnings"], narrow_zone_warning]

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
