from __future__ import annotations

import pandas as pd

ACTIVE_SIGNALS = {"Hold / Buy Candidate", "Watch", "Reduce / Avoid"}
SIGNAL_DISTRIBUTION_ORDER = [
    "Hold / Buy Candidate",
    "Watch",
    "Reduce / Avoid",
    "Cash Reserve / No Signal",
    "Insufficient Data",
]


def _get_signal_series(signals: pd.DataFrame) -> pd.Series:
    if "signal" not in signals:
        return pd.Series(dtype=object)
    return signals["signal"].astype(str)


def calculate_watchlist_health_score(signals: pd.DataFrame) -> dict[str, object]:
    signal_series = _get_signal_series(signals)
    if "ticker" in signals:
        ticker_series = signals["ticker"].astype(str).str.upper()
    else:
        ticker_series = pd.Series(index=signals.index, dtype=str)

    active_mask = signal_series.isin(ACTIVE_SIGNALS)
    if not ticker_series.empty:
        active_mask = active_mask & (ticker_series != "CASH")

    active_tradeable_count = int(active_mask.sum())
    hold_buy_count = int((signal_series == "Hold / Buy Candidate").loc[active_mask].sum())
    score = 0 if active_tradeable_count == 0 else round(100 * hold_buy_count / active_tradeable_count)

    if score <= 39:
        label = "Risk-Off / Caution"
    elif score <= 69:
        label = "Neutral / Selective"
    else:
        label = "Risk-On / Constructive"

    return {
        "score": score,
        "label": label,
        "hold_buy_count": hold_buy_count,
        "active_tradeable_count": active_tradeable_count,
    }


def calculate_action_urgency_score(signals: pd.DataFrame) -> dict[str, object]:
    signal_series = _get_signal_series(signals)
    reduce_avoid_count = int((signal_series == "Reduce / Avoid").sum())
    watch_count = int((signal_series == "Watch").sum())
    score = min(100, reduce_avoid_count * 40 + watch_count * 15)

    if score <= 24:
        label = "Low"
    elif score <= 59:
        label = "Review"
    else:
        label = "High Attention"

    return {
        "score": score,
        "label": label,
        "reduce_avoid_count": reduce_avoid_count,
        "watch_count": watch_count,
    }


def build_signal_distribution(signals: pd.DataFrame) -> pd.DataFrame:
    signal_series = _get_signal_series(signals)
    counts = signal_series.value_counts()
    rows = [
        {"signal_group": label, "count": int(counts.get(label, 0))}
        for label in SIGNAL_DISTRIBUTION_ORDER
    ]
    return pd.DataFrame(rows)


def build_allocation_by_bucket(watchlist: pd.DataFrame) -> pd.DataFrame:
    if "bucket" not in watchlist or "target_weight" not in watchlist:
        return pd.DataFrame(columns=["bucket", "target_weight"])

    working = watchlist[["bucket", "target_weight"]].copy()
    working["target_weight"] = pd.to_numeric(working["target_weight"], errors="coerce")
    working = working.dropna(subset=["bucket", "target_weight"])
    if working.empty:
        return pd.DataFrame(columns=["bucket", "target_weight"])

    allocation = (
        working.groupby("bucket", as_index=False)["target_weight"]
        .sum()
        .sort_values("target_weight", ascending=False)
    )
    allocation["target_weight"] = allocation["target_weight"].astype(float)
    return allocation


def build_entry_zone_dataframe(entry_guidance: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for zone_name, key in [
        ("Aggressive Bid Zone", "aggressive_bid_zone_bounds"),
        ("Normal Bid Zone", "normal_bid_zone_bounds"),
        ("Conservative Bid Zone", "conservative_bid_zone_bounds"),
    ]:
        bounds = entry_guidance.get(key) or {}
        low = bounds.get("low")
        high = bounds.get("high")
        if low is None or high is None:
            continue
        norm_low = min(float(low), float(high))
        norm_high = max(float(low), float(high))
        rows.append(
            {
                "label": zone_name,
                "display_low": norm_low,
                "display_high": norm_high,
                "was_normalized": bool(float(low) > float(high)),
            }
        )

    return pd.DataFrame(rows)
