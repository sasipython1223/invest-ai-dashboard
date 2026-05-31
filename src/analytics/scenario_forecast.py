from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
MIN_RETURN_OBSERVATIONS = 60
DEFAULT_HORIZONS: dict[str, int] = {"1M": 21, "3M": 63, "6M": 126}


def _normalize_price_series(price_series: pd.Series) -> pd.Series:
    prices = pd.to_numeric(pd.Series(price_series), errors="coerce").dropna()
    prices = prices[prices > 0]
    if len(prices) > TRADING_DAYS_PER_YEAR + 1:
        prices = prices.iloc[-(TRADING_DAYS_PER_YEAR + 1) :]
    return prices


def calculate_log_returns(price_series: pd.Series) -> pd.Series:
    prices = _normalize_price_series(price_series)
    if prices.empty:
        return pd.Series(dtype=float)
    return np.log(prices / prices.shift(1)).dropna()


def calculate_annualized_return_volatility(log_returns: pd.Series) -> dict[str, float]:
    returns = pd.to_numeric(pd.Series(log_returns), errors="coerce").dropna()
    if returns.empty:
        return {"annual_return": float("nan"), "annual_volatility": float("nan"), "observations": 0}

    mean_daily_return = float(returns.mean())
    std_daily_return = float(returns.std())
    return {
        "annual_return": mean_daily_return * TRADING_DAYS_PER_YEAR,
        "annual_volatility": std_daily_return * np.sqrt(TRADING_DAYS_PER_YEAR),
        "observations": float(len(returns)),
    }


def assign_volatility_risk_status(annual_volatility: float) -> str:
    if pd.isna(annual_volatility):
        return "N/A"
    if annual_volatility < 0.15:
        return "Low"
    if annual_volatility <= 0.30:
        return "Moderate"
    if annual_volatility <= 0.50:
        return "Elevated"
    return "High"


def build_volatility_cone(
    price_series: pd.Series,
    current_value: float | None = None,
    horizon_days: int = 126,
    steps: int | None = None,
) -> pd.DataFrame:
    columns = [
        "step",
        "horizon_days",
        "expected",
        "normal_upper",
        "normal_lower",
        "extreme_upper",
        "extreme_lower",
    ]
    if horizon_days <= 0:
        return pd.DataFrame(columns=columns)

    log_returns = calculate_log_returns(price_series)
    if len(log_returns) < MIN_RETURN_OBSERVATIONS:
        return pd.DataFrame(columns=columns)

    stats = calculate_annualized_return_volatility(log_returns)
    annual_return = stats["annual_return"]
    annual_volatility = stats["annual_volatility"]
    if pd.isna(annual_return) or pd.isna(annual_volatility):
        return pd.DataFrame(columns=columns)

    prices = _normalize_price_series(price_series)
    if prices.empty:
        return pd.DataFrame(columns=columns)
    current = float(prices.iloc[-1]) if current_value is None else float(current_value)
    if current <= 0:
        return pd.DataFrame(columns=columns)

    effective_steps = int(horizon_days if steps is None else steps)
    effective_steps = max(effective_steps, 1)
    step_days = np.unique(np.round(np.linspace(0, horizon_days, effective_steps + 1)).astype(int))
    if len(step_days) == 0 or step_days[0] != 0:
        step_days = np.insert(step_days, 0, 0)

    t = step_days / TRADING_DAYS_PER_YEAR
    drift = annual_return * t
    spread = annual_volatility * np.sqrt(t)

    cone = pd.DataFrame(
        {
            "step": step_days,
            "horizon_days": int(horizon_days),
            "expected": current * np.exp(drift),
            "normal_upper": current * np.exp(drift + spread),
            "normal_lower": current * np.exp(drift - spread),
            "extreme_upper": current * np.exp(drift + 2 * spread),
            "extreme_lower": current * np.exp(drift - 2 * spread),
        }
    )
    return cone[columns]


def build_scenario_summary(
    price_series: pd.Series,
    horizons: dict[str, int] | None = None,
) -> pd.DataFrame:
    columns = [
        "horizon",
        "best_2sd",
        "normal_upper_1sd",
        "expected",
        "normal_lower_1sd",
        "worst_2sd",
        "annual_return",
        "annual_volatility",
    ]
    log_returns = calculate_log_returns(price_series)
    if len(log_returns) < MIN_RETURN_OBSERVATIONS:
        return pd.DataFrame(columns=columns)

    stats = calculate_annualized_return_volatility(log_returns)
    annual_return = stats["annual_return"]
    annual_volatility = stats["annual_volatility"]
    if pd.isna(annual_return) or pd.isna(annual_volatility):
        return pd.DataFrame(columns=columns)

    prices = _normalize_price_series(price_series)
    if prices.empty:
        return pd.DataFrame(columns=columns)
    current_value = float(prices.iloc[-1])
    if current_value <= 0:
        return pd.DataFrame(columns=columns)

    summary_rows: list[dict[str, float | str]] = []
    for horizon_label, horizon_days in (horizons or DEFAULT_HORIZONS).items():
        if horizon_days <= 0:
            continue
        t = float(horizon_days) / TRADING_DAYS_PER_YEAR
        drift = annual_return * t
        spread = annual_volatility * np.sqrt(t)
        summary_rows.append(
            {
                "horizon": horizon_label,
                "best_2sd": current_value * np.exp(drift + 2 * spread),
                "normal_upper_1sd": current_value * np.exp(drift + spread),
                "expected": current_value * np.exp(drift),
                "normal_lower_1sd": current_value * np.exp(drift - spread),
                "worst_2sd": current_value * np.exp(drift - 2 * spread),
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
            }
        )

    return pd.DataFrame(summary_rows, columns=columns)
