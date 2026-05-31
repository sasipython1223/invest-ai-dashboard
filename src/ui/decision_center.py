from __future__ import annotations

import pandas as pd

from src.ui.charts import calculate_action_urgency_score, calculate_watchlist_health_score

_EMPTY_GROUPS = {
    "hold_buy": [],
    "watch": [],
    "reduce_avoid": [],
    "cash": [],
}


def group_tickers_by_signal(signals: pd.DataFrame) -> dict[str, list[str]]:
    if "ticker" not in signals:
        return {key: [] for key in _EMPTY_GROUPS}

    ticker_series = signals["ticker"].astype(str).str.upper()
    signal_series = signals["signal"].astype(str) if "signal" in signals else pd.Series(dtype=object)

    grouped = {key: [] for key in _EMPTY_GROUPS}
    for ticker, signal in zip(ticker_series.tolist(), signal_series.tolist()):
        if ticker == "CASH":
            grouped["cash"].append(ticker)
        elif signal == "Hold / Buy Candidate":
            grouped["hold_buy"].append(ticker)
        elif signal == "Watch":
            grouped["watch"].append(ticker)
        elif signal == "Reduce / Avoid":
            grouped["reduce_avoid"].append(ticker)

    return grouped


def build_ticker_name_lookup(watchlist: pd.DataFrame) -> dict[str, str]:
    if "ticker" not in watchlist or "name" not in watchlist:
        return {}
    lookup: dict[str, str] = {}
    for _, row in watchlist[["ticker", "name"]].dropna().iterrows():
        ticker = str(row["ticker"]).strip().upper()
        if not ticker:
            continue
        lookup[ticker] = str(row["name"]).strip()
    return lookup


def format_ticker_list(tickers: list[str], name_lookup: dict[str, str]) -> str:
    if not tickers:
        return "None"
    formatted: list[str] = []
    for ticker in tickers:
        name = name_lookup.get(ticker, "")
        formatted.append(f"{ticker} — {name}" if name else ticker)
    return ", ".join(formatted)


def build_gauge_explanation(
    signals: pd.DataFrame,
    health_score: int,
    urgency_score: int,
    reserve_target_weight: float,
) -> dict[str, object]:
    grouped = group_tickers_by_signal(signals)
    health_summary = calculate_watchlist_health_score(signals)
    urgency_summary = calculate_action_urgency_score(signals)
    reduce_list = ", ".join(grouped["reduce_avoid"]) if grouped["reduce_avoid"] else "None"

    health_explanation = (
        f"{health_score} = {health_summary['label']} watchlist condition. "
        f"Calculated from {len(grouped['hold_buy'])} Hold / Buy candidates, "
        f"{len(grouped['watch'])} Watch items, {len(grouped['reduce_avoid'])} Reduce / Avoid items, "
        f"and {len(grouped['cash'])} Cash reserve item{'s' if len(grouped['cash']) != 1 else ''}."
    )
    urgency_explanation = (
        f"{urgency_score} = {urgency_summary['label']} required. "
        f"Driven mainly by {len(grouped['reduce_avoid'])} Reduce / Avoid holding"
        f"{'s' if len(grouped['reduce_avoid']) != 1 else ''}: {reduce_list}."
    )

    reduce_context = (
        f"{reduce_list} is flagged Reduce / Avoid"
        if reduce_list != "None"
        else "there are no Reduce / Avoid tickers currently"
    )
    interpretation = (
        "Watchlist is constructive, but action urgency is Review because "
        f"{reduce_context}. New deployment should remain cautious. "
        "Prefer balanced deployment, keep cash reserve, and manually review entry zones before placing any limit order."
    )

    return {
        "health_explanation": health_explanation,
        "urgency_explanation": urgency_explanation,
        "urgency_drivers": [
            f"Reduce / Avoid: {reduce_list}",
            f"Cash reserve target: {reserve_target_weight:.2f}%",
            "No automatic execution",
        ],
        "grouped_tickers": grouped,
        "interpretation": interpretation,
        "scoring_logic_lines": [
            "Watchlist Health = 100 × Hold / Buy candidates ÷ active tradeable instruments (excluding CASH).",
            "Action Urgency = min(100, 40 × Reduce / Avoid count + 15 × Watch count).",
            "More Hold / Buy candidates lift health; more Watch or Reduce / Avoid items increase urgency.",
        ],
    }


def get_current_reserve_target(watchlist: pd.DataFrame) -> float:
    if "ticker" not in watchlist or "target_weight" not in watchlist:
        return 0.0
    cash_rows = watchlist.loc[watchlist["ticker"].astype(str).str.upper() == "CASH", "target_weight"]
    if cash_rows.empty:
        return 0.0
    return float(pd.to_numeric(cash_rows.iloc[0], errors="coerce") or 0.0)


def calculate_cash_reserve_scenario(
    review_amount: float,
    current_reserve_pct: float,
    test_reserve_pct: float,
) -> pd.DataFrame:
    def _row(label: str, reserve_pct: float) -> dict[str, float | str]:
        keep_cash = round(float(review_amount) * float(reserve_pct) / 100.0, 2)
        deploy_amount = round(max(0.0, float(review_amount) - keep_cash), 2)
        return {
            "scenario": label,
            "reserve_target_pct": float(reserve_pct),
            "keep_as_cash": keep_cash,
            "deploy_for_review": deploy_amount,
        }

    return pd.DataFrame(
        [
            _row("Current target", current_reserve_pct),
            _row("Test target", test_reserve_pct),
        ]
    )
