from __future__ import annotations

import math

import pandas as pd

_DEPLOY_RATIOS = {
    "Conservative": 0.4,
    "Balanced": 0.6,
    "Opportunistic": 0.8,
}

_ELIGIBLE_SIGNALS = {"Hold / Buy Candidate", "Watch"}


def _normalize_style(style: str) -> str:
    return style if style in _DEPLOY_RATIOS else "Balanced"


def _risk_is_acceptable(risk: dict) -> bool:
    status = str(risk.get("status", "")).upper()
    return bool(status) and "HIGH" not in status and status != "REVIEW"


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _get_currency(watchlist: pd.DataFrame) -> str:
    if watchlist.empty:
        return "SGD"

    cash_rows = watchlist.loc[watchlist["ticker"].astype(str).str.upper() == "CASH"]
    if not cash_rows.empty and "currency" in cash_rows.columns:
        cash_currency = str(cash_rows.iloc[0].get("currency", "")).strip()
        if cash_currency:
            return cash_currency

    if "currency" in watchlist.columns:
        first_currency = str(watchlist.iloc[0].get("currency", "")).strip()
        if first_currency:
            return first_currency

    return "SGD"


def _candidate_priority(row: pd.Series) -> tuple[int, float, str]:
    bucket = str(row.get("bucket", "")).strip().lower()
    asset_type = str(row.get("type", "")).strip().upper()
    name = str(row.get("name", "")).strip().lower()
    ticker = str(row.get("ticker", "")).strip().upper()
    target_weight = _to_float(row.get("target_weight"), 0.0)

    broad_keywords = ("all-world", "s&p", "msci", "index", "etf")
    is_core_broad_etf = bucket == "core" and asset_type == "ETF" and any(k in name for k in broad_keywords)

    if is_core_broad_etf:
        rank = 0
    elif bucket == "core" and asset_type == "ETF":
        rank = 1
    elif bucket == "singapore":
        rank = 2
    elif bucket == "tactical":
        rank = 4
    else:
        rank = 3

    return (rank, -target_weight, ticker)


def _build_tranches(amount: float) -> tuple[float, float, float]:
    tranche_1 = round(amount * 0.5, 2)
    tranche_2 = round(amount * (1 / 3), 2)
    tranche_3 = round(amount - tranche_1 - tranche_2, 2)
    return tranche_1, tranche_2, tranche_3


def build_cash_deployment_plan(
    amount: float,
    watchlist: pd.DataFrame,
    signals: pd.DataFrame,
    risk: dict,
    deployment_style: str,
    include_tactical: bool = False,
) -> dict[str, object]:
    warnings: list[str] = []
    excluded_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []

    style = _normalize_style(deployment_style)
    if style != deployment_style:
        warnings.append("Invalid deployment style provided; defaulted to Balanced.")

    safe_amount = _to_float(amount, default=0.0)
    currency = _get_currency(watchlist)

    if safe_amount <= 0:
        warnings.append("Cash amount must be greater than 0 for deployment review.")
        return {
            "amount": round(max(0.0, safe_amount), 2),
            "currency": currency,
            "deployment_style": style,
            "deploy_now_amount": 0.0,
            "keep_as_cash": round(max(0.0, safe_amount), 2),
            "allocation_rows": allocation_rows,
            "excluded_rows": excluded_rows,
            "warnings": warnings,
            "manual_review_required": True,
        }

    deploy_ratio = _DEPLOY_RATIOS[style]
    if style == "Opportunistic" and not _risk_is_acceptable(risk):
        deploy_ratio = _DEPLOY_RATIOS["Balanced"]
        style = "Balanced"
        warnings.append("Risk status not acceptable for Opportunistic deployment; fallback to Balanced.")

    deploy_now_amount = round(safe_amount * deploy_ratio, 2)
    keep_as_cash = round(safe_amount - deploy_now_amount, 2)

    if watchlist.empty or signals.empty:
        warnings.append("Watchlist or signal data is unavailable; no allocation rows generated.")
        return {
            "amount": round(safe_amount, 2),
            "currency": currency,
            "deployment_style": style,
            "deploy_now_amount": deploy_now_amount,
            "keep_as_cash": keep_as_cash,
            "allocation_rows": allocation_rows,
            "excluded_rows": excluded_rows,
            "warnings": warnings,
            "manual_review_required": True,
        }

    signal_cols = ["ticker", "signal"]
    merged = watchlist.merge(signals[signal_cols], on="ticker", how="left", suffixes=("", "_signal"))

    eligible_rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        bucket = str(row.get("bucket", "")).strip().lower()
        signal = str(row.get("signal", "")).strip()
        asset_type = str(row.get("type", "")).strip().upper()

        exclusion_reason = ""
        if ticker == "CASH" or bucket == "cash" or asset_type == "CASH":
            exclusion_reason = "Cash reserve is excluded from deployment candidates."
        elif signal == "Reduce / Avoid":
            exclusion_reason = "Reduce / Avoid signal is excluded from deployment candidates."
        elif bucket == "tactical" and not include_tactical:
            exclusion_reason = "Tactical bucket excluded unless include_tactical is enabled."
        elif signal not in _ELIGIBLE_SIGNALS:
            exclusion_reason = "Signal is not eligible for deployment review."

        if exclusion_reason:
            excluded_rows.append(
                {
                    "ticker": ticker,
                    "bucket": bucket,
                    "signal": signal,
                    "reason": exclusion_reason,
                }
            )
            continue

        candidate = row.to_dict()
        candidate["target_weight"] = max(_to_float(candidate.get("target_weight"), 0.0), 0.0)
        eligible_rows.append(candidate)

    if not eligible_rows or deploy_now_amount <= 0:
        warnings.append("No eligible deployment candidates after rule filters.")
        return {
            "amount": round(safe_amount, 2),
            "currency": currency,
            "deployment_style": style,
            "deploy_now_amount": deploy_now_amount,
            "keep_as_cash": keep_as_cash,
            "allocation_rows": allocation_rows,
            "excluded_rows": excluded_rows,
            "warnings": warnings,
            "manual_review_required": True,
        }

    eligible_df = pd.DataFrame(eligible_rows)
    eligible_df = eligible_df.iloc[
        sorted(range(len(eligible_df)), key=lambda i: _candidate_priority(eligible_df.iloc[i]))
    ]

    weights = eligible_df["target_weight"].astype(float)
    if float(weights.sum()) <= 0:
        warnings.append("Eligible target weights are missing; applying equal-weight distribution.")
        weights = pd.Series(1.0, index=eligible_df.index)

    weighted_amounts = deploy_now_amount * (weights / float(weights.sum()))
    running_total = 0.0
    for position, (_, row) in enumerate(eligible_df.iterrows()):
        if position == len(eligible_df) - 1:
            suggested_amount = round(deploy_now_amount - running_total, 2)
        else:
            suggested_amount = round(float(weighted_amounts.iloc[position]), 2)
            running_total += suggested_amount

        tranche_1, tranche_2, tranche_3 = _build_tranches(suggested_amount)
        allocation_rows.append(
            {
                "ticker": str(row.get("ticker", "")).upper(),
                "bucket": str(row.get("bucket", "")),
                "signal": str(row.get("signal", "")),
                "suggested_amount": suggested_amount,
                "tranche_1": tranche_1,
                "tranche_2": tranche_2,
                "tranche_3": tranche_3,
                "execution_note": "Review Entry Guidance and use manual limit order only.",
            }
        )

    return {
        "amount": round(safe_amount, 2),
        "currency": currency,
        "deployment_style": style,
        "deploy_now_amount": deploy_now_amount,
        "keep_as_cash": keep_as_cash,
        "allocation_rows": allocation_rows,
        "excluded_rows": excluded_rows,
        "warnings": warnings,
        "manual_review_required": True,
    }
