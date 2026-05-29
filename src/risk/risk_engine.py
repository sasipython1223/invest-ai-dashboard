from __future__ import annotations

import pandas as pd

from src.risk.exposure_checks import check_target_weight_sum


def evaluate_risk(watchlist: pd.DataFrame, signals: pd.DataFrame) -> dict[str, object]:
    exposure = check_target_weight_sum(watchlist)

    reduce_count = int((signals["signal"] == "Reduce / Avoid").sum()) if not signals.empty else 0
    status = "OK"
    alerts: list[str] = []

    if not exposure["ok"]:
        status = "REVIEW"
        alerts.append(
            f"Target weights sum to {exposure['total_target_weight']:.2f}%, expected 100.00%."
        )

    if reduce_count > 0:
        status = "REVIEW"
        alerts.append(f"{reduce_count} holdings flagged as Reduce / Avoid.")

    if not alerts:
        alerts.append("No rule-based risk alerts.")

    return {
        "status": status,
        "alerts": alerts,
        "exposure": exposure,
    }
