import pandas as pd

TRADING_DAYS_PER_MONTH = 21


def compute_momentum(series: pd.Series, months: int) -> float | None:
    if series is None or len(series) < (months * TRADING_DAYS_PER_MONTH + 1):
        return None
    lookback = months * TRADING_DAYS_PER_MONTH
    latest = float(series.iloc[-1])
    past = float(series.iloc[-(lookback + 1)])
    if past == 0:
        return None
    return latest / past - 1
