from __future__ import annotations

import pandas as pd

MANUAL_REVIEW_NOTE = "Manual review required. No trades are executed by this dashboard."
_SIGNAL_GROUPS = {
    "Hold / Buy Candidate": "hold_buy_candidates",
    "Watch": "watch_items",
    "Reduce / Avoid": "reduce_avoid_items",
}


def count_signals(signals: pd.DataFrame) -> dict[str, int]:
    signal_series = signals["signal"] if "signal" in signals else pd.Series(dtype=object)
    return {
        "total_instruments": int(len(signals)),
        **{
            key: int((signal_series == label).sum())
            for label, key in _SIGNAL_GROUPS.items()
        },
    }



def build_action_items(signals: pd.DataFrame, risk: dict[str, object]) -> list[str]:
    items: list[str] = []
    ticker_series = signals["ticker"].astype(str) if "ticker" in signals else pd.Series(dtype=str)
    signal_series = signals["signal"] if "signal" in signals else pd.Series(dtype=object)

    for label, prefix in (
        ("Reduce / Avoid", "Review Reduce / Avoid items"),
        ("Watch", "Watch items"),
        ("Hold / Buy Candidate", "Hold / Buy candidates"),
    ):
        tickers = ticker_series.loc[signal_series == label].tolist()
        if tickers:
            items.append(f"{prefix}: {', '.join(tickers)}. {MANUAL_REVIEW_NOTE}")

    reserve_target_weight = float(risk.get("reserve_target_weight", 0.0) or 0.0)
    items.append(f"Cash reserve: {reserve_target_weight:.2f}% target weight.")

    if not any(signal_series.isin(_SIGNAL_GROUPS)):
        items.insert(0, "No urgent signal changes today.")

    return [f"{index}. {item}" for index, item in enumerate(items, start=1)]
