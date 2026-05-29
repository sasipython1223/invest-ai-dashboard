from __future__ import annotations

import pandas as pd

from src.signals.momentum_signal import compute_momentum
from src.signals.trend_signal import compute_sma


SIGNAL_INSUFFICIENT = "Insufficient Data"


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
        metrics = evaluate_signal(prices.get(ticker, pd.Series(dtype=float)))
        rows.append(
            {
                "ticker": ticker,
                "name": row.get("name", ""),
                "bucket": row.get("bucket", ""),
                **metrics,
            }
        )

    return pd.DataFrame(rows)
