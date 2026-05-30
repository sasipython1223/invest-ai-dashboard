from __future__ import annotations

import pandas as pd


def resolve_price_history(
    prices: dict[str, pd.Series],
    ticker: str,
    data_ticker: str | None = None,
) -> pd.Series:
    """Resolve price history by display ticker, then data_ticker fallback.

    First tries the display ticker key (e.g. ``VWRA``); if the series is
    absent or empty, falls back to the data_ticker key (e.g. ``VWRA.L``).
    Returns an empty Series when neither key yields usable history.
    Never calls external APIs.
    """
    series = prices.get(ticker)
    if series is not None:
        cleaned = pd.Series(series, dtype=float).dropna()
        if not cleaned.empty:
            return cleaned
    if data_ticker and data_ticker != ticker:
        series = prices.get(data_ticker)
        if series is not None:
            cleaned = pd.Series(series, dtype=float).dropna()
            if not cleaned.empty:
                return cleaned
    return pd.Series(dtype=float)


def build_indexed_price_series(price_history: pd.Series) -> pd.Series:
    history = pd.Series(price_history, dtype=float).dropna()
    if history.empty:
        return pd.Series(dtype=float)

    start_price = float(history.iloc[0])
    if start_price <= 0:
        return pd.Series(dtype=float)

    return history / start_price * 100.0


def build_weighted_portfolio_index(
    prices: dict[str, pd.Series],
    watchlist: pd.DataFrame,
) -> pd.Series:
    if watchlist.empty or "ticker" not in watchlist or "target_weight" not in watchlist:
        return pd.Series(dtype=float)

    indexed_series_by_ticker: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}

    for _, row in watchlist.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        if ticker == "CASH" or ticker == "":
            continue

        target_weight = pd.to_numeric(row.get("target_weight"), errors="coerce")
        if pd.isna(target_weight) or float(target_weight) <= 0:
            continue

        data_ticker = str(row.get("data_ticker", "") or "").strip() or None
        indexed_series = build_indexed_price_series(
            resolve_price_history(prices, ticker, data_ticker)
        )
        if len(indexed_series) < 2:
            continue

        indexed_series_by_ticker[ticker] = indexed_series
        weights[ticker] = float(target_weight)

    if not indexed_series_by_ticker:
        return pd.Series(dtype=float)

    indexed_frame = pd.concat(indexed_series_by_ticker, axis=1, join="inner")
    if indexed_frame.empty:
        return pd.Series(dtype=float)

    weight_series = pd.Series(weights, dtype=float).reindex(indexed_frame.columns)
    weight_sum = float(weight_series.sum())
    if weight_sum <= 0:
        return pd.Series(dtype=float)

    normalized_weights = weight_series / weight_sum
    weighted_index = indexed_frame.mul(normalized_weights, axis=1).sum(axis=1)
    return build_indexed_price_series(weighted_index)


def build_drawdown_series(index_series: pd.Series) -> pd.Series:
    indexed = pd.Series(index_series, dtype=float).dropna()
    if indexed.empty:
        return pd.Series(dtype=float)

    rolling_peak = indexed.cummax()
    drawdown = indexed / rolling_peak - 1.0
    drawdown[drawdown.abs() < 1e-12] = 0.0
    return drawdown


def rebase_comparison_frame(compare_df: pd.DataFrame) -> pd.DataFrame:
    comparison = pd.DataFrame(compare_df, dtype=float).dropna()
    if comparison.empty:
        return pd.DataFrame(dtype=float)

    return comparison / comparison.iloc[0] * 100.0


def build_ticker_trend_dataframe(price_history: pd.Series) -> pd.DataFrame:
    history = pd.Series(price_history, dtype=float).dropna()
    if history.empty:
        return pd.DataFrame(columns=["price", "sma_20", "sma_50", "sma_200"])

    trend_df = pd.DataFrame({"price": history})
    trend_df["sma_20"] = trend_df["price"].rolling(window=20, min_periods=20).mean()
    trend_df["sma_50"] = trend_df["price"].rolling(window=50, min_periods=50).mean()
    trend_df["sma_200"] = trend_df["price"].rolling(window=200, min_periods=200).mean()
    return trend_df


def build_return_summary(
    signals: pd.DataFrame,
    prices: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    columns = ["ticker", "name", "bucket", "signal", "momentum_3m", "momentum_6m"]
    if signals.empty:
        return pd.DataFrame(columns=[*columns, "return_1m"])

    summary = pd.DataFrame(index=signals.index)
    for column in columns:
        summary[column] = signals[column] if column in signals else None
    if "data_ticker" in signals.columns:
        summary["data_ticker"] = signals["data_ticker"]

    summary["momentum_3m"] = pd.to_numeric(summary["momentum_3m"], errors="coerce")
    summary["momentum_6m"] = pd.to_numeric(summary["momentum_6m"], errors="coerce")

    returns_1m: list[float | None] = []
    for _, row in summary.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        if not prices or ticker == "":
            returns_1m.append(None)
            continue

        data_ticker = str(row.get("data_ticker", "") or "").strip() or None
        history = resolve_price_history(prices, ticker, data_ticker)
        if len(history) < 22:
            returns_1m.append(None)
            continue

        start = float(history.iloc[-22])
        latest = float(history.iloc[-1])
        if start <= 0:
            returns_1m.append(None)
            continue
        returns_1m.append(latest / start - 1.0)

    summary["return_1m"] = returns_1m
    return summary
