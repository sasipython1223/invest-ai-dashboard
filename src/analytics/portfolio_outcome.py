from __future__ import annotations

import math

import pandas as pd

from src.analytics.scenario_forecast import build_scenario_summary

_ALLOCATION_COLUMNS = ["ticker", "name", "signal", "bucket", "allocation_pct", "target_weight", "data_ticker"]
_CLOSE_COLUMNS = ("Close", "Adj Close", "close", "adj_close")
TACTICAL_REVIEW_MODE_MULTIPLIER = 0.5


def _parse_float_with_default(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _normalize_weights(weights: pd.Series) -> pd.Series:
    clean = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    total = float(clean.sum())
    if total <= 0 and len(clean) > 0:
        clean = pd.Series(1.0, index=clean.index, dtype=float)
        total = float(clean.sum())
    if total <= 0:
        return pd.Series(dtype=float)
    return clean / total * 100.0


def _extract_and_clean_price_series(price_history: object) -> pd.Series:
    if isinstance(price_history, pd.DataFrame):
        for col in _CLOSE_COLUMNS:
            if col in price_history.columns:
                return pd.to_numeric(price_history[col], errors="coerce").dropna()
        if price_history.shape[1] == 1:
            return pd.to_numeric(price_history.iloc[:, 0], errors="coerce").dropna()
        return pd.Series(dtype=float)

    if isinstance(price_history, pd.Series):
        return pd.to_numeric(price_history, errors="coerce").dropna()

    try:
        return pd.to_numeric(pd.Series(price_history), errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)


def _resolve_price_history(
    prices: dict[str, pd.Series],
    ticker: str,
    data_ticker: str | None = None,
) -> pd.Series:
    value = prices.get(ticker)
    if value is not None:
        cleaned = _extract_and_clean_price_series(value)
        if not cleaned.empty:
            return cleaned
    if data_ticker and data_ticker != ticker:
        value = prices.get(data_ticker)
        if value is not None:
            cleaned = _extract_and_clean_price_series(value)
            if not cleaned.empty:
                return cleaned
    return pd.Series(dtype=float)


def build_default_allocation(
    signals: pd.DataFrame,
    watchlist: pd.DataFrame,
    include_tactical: bool = False,
    risk_status: str | None = None,
) -> pd.DataFrame:
    if watchlist.empty or signals.empty:
        return pd.DataFrame(columns=_ALLOCATION_COLUMNS)

    signal_cols = [col for col in ("ticker", "signal") if col in signals.columns]
    if "ticker" not in signal_cols:
        return pd.DataFrame(columns=_ALLOCATION_COLUMNS)
    merged = watchlist.merge(signals[signal_cols], on="ticker", how="left")

    rows: list[dict[str, object]] = []
    review_mode = str(risk_status or "").strip().upper() == "REVIEW"
    for _, row in merged.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        signal = str(row.get("signal", "")).strip()
        bucket = str(row.get("bucket", "")).strip().lower()
        asset_type = str(row.get("type", "")).strip().upper()

        if ticker == "" or ticker == "CASH":
            continue
        if bucket == "cash" or asset_type == "CASH":
            continue
        if signal != "Hold / Buy Candidate":
            continue
        if bucket == "tactical" and not include_tactical:
            continue

        weight = max(_parse_float_with_default(row.get("target_weight"), 0.0), 0.0)
        if review_mode and bucket == "tactical":
            weight *= TACTICAL_REVIEW_MODE_MULTIPLIER

        rows.append(
            {
                "ticker": ticker,
                "name": str(row.get("name", "")).strip(),
                "signal": signal,
                "bucket": str(row.get("bucket", "")).strip(),
                "target_weight": weight,
                "data_ticker": str(row.get("data_ticker", "") or "").strip() or None,
            }
        )

    if not rows:
        return pd.DataFrame(columns=_ALLOCATION_COLUMNS)

    allocation = pd.DataFrame(rows)
    allocation["allocation_pct"] = _normalize_weights(allocation["target_weight"])
    return allocation[_ALLOCATION_COLUMNS]


def normalize_allocations(allocation_df: pd.DataFrame) -> pd.DataFrame:
    if allocation_df.empty or "allocation_pct" not in allocation_df.columns:
        return allocation_df.copy()

    normalized = allocation_df.copy()
    normalized["allocation_pct"] = _normalize_weights(normalized["allocation_pct"])
    return normalized


def calculate_ticker_outcome(
    ticker: str,
    allocated_amount: float,
    price_history: pd.Series,
    horizon_label: str,
) -> dict[str, float | str]:
    base_row: dict[str, float | str] = {
        "ticker": ticker,
        "status": "skipped_insufficient_history",
        "allocated_amount": float(allocated_amount),
        "best_value": 0.0,
        "normal_upper_value": 0.0,
        "expected_value": 0.0,
        "normal_lower_value": 0.0,
        "worst_value": 0.0,
        "expected_gain_loss": -float(allocated_amount),
        "worst_loss": -float(allocated_amount),
    }
    if allocated_amount <= 0:
        base_row["status"] = "skipped_non_positive_allocation"
        base_row["expected_gain_loss"] = 0.0
        base_row["worst_loss"] = 0.0
        return base_row

    history = _extract_and_clean_price_series(price_history)
    summary_df = build_scenario_summary(history)
    if history.empty or summary_df.empty:
        return base_row

    horizon_row = summary_df.loc[summary_df["horizon"] == horizon_label]
    if horizon_row.empty:
        base_row["status"] = "skipped_missing_horizon"
        return base_row

    current_price = float(history.iloc[-1])
    if current_price <= 0:
        base_row["status"] = "skipped_invalid_current_price"
        return base_row

    row = horizon_row.iloc[0]
    amount_per_price_unit = float(allocated_amount) / current_price
    scenario_amounts = {
        "best_value": amount_per_price_unit * float(row["best_2sd"]),
        "normal_upper_value": amount_per_price_unit * float(row["normal_upper_1sd"]),
        "expected_value": amount_per_price_unit * float(row["expected"]),
        "normal_lower_value": amount_per_price_unit * float(row["normal_lower_1sd"]),
        "worst_value": amount_per_price_unit * float(row["worst_2sd"]),
    }

    return {
        "ticker": ticker,
        "status": "ok",
        "allocated_amount": float(allocated_amount),
        **scenario_amounts,
        "expected_gain_loss": scenario_amounts["expected_value"] - float(allocated_amount),
        "worst_loss": scenario_amounts["worst_value"] - float(allocated_amount),
    }


def calculate_portfolio_outcome(
    allocation_df: pd.DataFrame,
    prices: dict[str, pd.Series],
    total_investment: float,
    horizon_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if allocation_df.empty:
        ticker_columns = [
            "ticker",
            "allocation_pct",
            "amount",
            "best_value",
            "normal_upper_value",
            "expected_value",
            "normal_lower_value",
            "worst_value",
            "expected_gain_loss",
            "worst_loss",
            "status",
        ]
        portfolio_columns = ["Scenario", "Ending Value", "Gain / Loss", "Return %"]
        return pd.DataFrame(columns=ticker_columns), pd.DataFrame(columns=portfolio_columns)

    rows: list[dict[str, object]] = []
    for _, row in allocation_df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        allocation_pct = _parse_float_with_default(row.get("allocation_pct"), 0.0)
        amount = float(total_investment) * allocation_pct / 100.0
        data_ticker = str(row.get("data_ticker", "") or "").strip() or None
        history = _resolve_price_history(prices, ticker, data_ticker)
        outcome = calculate_ticker_outcome(ticker, amount, history, horizon_label)
        rows.append(
            {
                "ticker": ticker,
                "name": row.get("name"),
                "signal": row.get("signal"),
                "bucket": row.get("bucket"),
                "allocation_pct": allocation_pct,
                "amount": amount,
                **outcome,
            }
        )

    ticker_df = pd.DataFrame(rows)
    if ticker_df.empty:
        portfolio_columns = ["Scenario", "Ending Value", "Gain / Loss", "Return %"]
        return ticker_df, pd.DataFrame(columns=portfolio_columns)

    invested_amount = float(ticker_df["amount"].sum())
    valid = ticker_df.loc[ticker_df["status"] == "ok"]
    totals = {
        "Best (+2 SD)": float(valid["best_value"].sum()),
        "Normal Upper (+1 SD)": float(valid["normal_upper_value"].sum()),
        "Likely / Expected": float(valid["expected_value"].sum()),
        "Normal Lower (-1 SD)": float(valid["normal_lower_value"].sum()),
        "Worst (-2 SD)": float(valid["worst_value"].sum()),
    }

    portfolio_rows = []
    for label, ending_value in totals.items():
        gain_loss = ending_value - invested_amount
        return_pct = gain_loss / invested_amount * 100.0 if invested_amount > 0 else 0.0
        portfolio_rows.append(
            {
                "Scenario": label,
                "Ending Value": ending_value,
                "Gain / Loss": gain_loss,
                "Return %": return_pct,
            }
        )

    return ticker_df, pd.DataFrame(portfolio_rows)
