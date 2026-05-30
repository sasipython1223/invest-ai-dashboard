from __future__ import annotations

import pandas as pd

from src.data.asset_classifier import is_tradeable_asset
from src.signals.momentum_signal import compute_momentum
from src.signals.trend_signal import compute_sma


SIGNAL_INSUFFICIENT = "Insufficient Data"
SIGNAL_CASH_RESERVE = "Cash Reserve / No Signal"
SIGNAL_CASH_REASON = "Non-tradeable reserve asset; no technical signal generated."


def evaluate_signal(price_series: pd.Series) -> dict[str, float | str | None]:
    latest_price = float(price_series.iloc[-1]) if not price_series.empty else None
    sma_200 = compute_sma(price_series, 200)
    momentum_6m = compute_momentum(price_series, 6)
    momentum_3m = compute_momentum(price_series, 3)

    if (
        latest_price is None
        or sma_200 is None
        or momentum_6m is None
        or momentum_3m is None
    ):
        signal = SIGNAL_INSUFFICIENT
        reason = "Not enough historical prices for full rule evaluation."
    elif latest_price > sma_200 and momentum_6m > 0:
        signal = "Hold / Buy Candidate"
        reason = "Price is above 200D SMA and 6M momentum is positive."
    elif latest_price > sma_200 and momentum_3m < 0:
        signal = "Watch"
        reason = "Price is above 200D SMA but 3M momentum is negative."
    elif latest_price < sma_200:
        signal = "Reduce / Avoid"
        reason = "Price is below 200D SMA."
    else:
        signal = "Watch"
        reason = "Mixed momentum conditions require manual review."

    return {
        "latest_price": latest_price,
        "sma_200": sma_200,
        "momentum_6m": momentum_6m,
        "momentum_3m": momentum_3m,
        "signal": signal,
        "signal_reason": reason,
    }


def run_signal_engine(watchlist: pd.DataFrame, prices: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for _, row in watchlist.iterrows():
        ticker = row["ticker"]
        if is_tradeable_asset(row):
            metrics = evaluate_signal(prices.get(ticker, pd.Series(dtype=float)))
        else:
            metrics = {
                "latest_price": None,
                "sma_200": None,
                "momentum_6m": None,
                "momentum_3m": None,
                "signal": SIGNAL_CASH_RESERVE,
                "signal_reason": SIGNAL_CASH_REASON,
            }
        rows.append(
            {
                "ticker": ticker,
                "name": row.get("name", ""),
                "type": row.get("type", ""),
                "bucket": row.get("bucket", ""),
                "data_ticker": row.get("data_ticker", ticker),
                **metrics,
            }
        )

    return pd.DataFrame(rows)
