from __future__ import annotations

import pandas as pd

from src.data.asset_classifier import is_tradeable_asset
from src.risk.exposure_checks import check_target_weight_sum


def evaluate_risk(watchlist: pd.DataFrame, signals: pd.DataFrame) -> dict[str, object]:
    exposure = check_target_weight_sum(watchlist)
    tradeable_mask = watchlist.apply(is_tradeable_asset, axis=1)
    tradeable_tickers = set(watchlist.loc[tradeable_mask, "ticker"].astype(str))
    reserve_target_weight = float(watchlist.loc[~tradeable_mask, "target_weight"].sum())

    if signals.empty:
        reduce_count = 0
    elif "ticker" in signals:
        reduce_count = int(
            signals.loc[
                signals["ticker"].astype(str).isin(tradeable_tickers)
                & (signals["signal"] == "Reduce / Avoid")
            ].shape[0]
        )
    else:
        reduce_count = int((signals["signal"] == "Reduce / Avoid").sum())
    status = "OK"
    alerts: list[str] = []

    if not exposure["ok"]:
        status = "REVIEW"
        alerts.append(
            f"Target weights sum to {exposure['total_target_weight']:.2f}%, expected 100.00%."
        )

    if reduce_count > 0:
        status = "REVIEW"
        holdings_label = "holding" if reduce_count == 1 else "holdings"
        alerts.append(f"{reduce_count} {holdings_label} flagged as Reduce / Avoid.")

    if not alerts:
        alerts.append("No rule-based risk alerts.")

    return {
        "status": status,
        "alerts": alerts,
        "exposure": exposure,
        "reserve_target_weight": reserve_target_weight,
    }
