import pandas as pd


def compute_sma(series: pd.Series, window: int = 200) -> float | None:
    if series is None or len(series) < window:
        return None
    return float(series.tail(window).mean())
