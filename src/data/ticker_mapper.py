from __future__ import annotations

import pandas as pd


def get_data_ticker(row: pd.Series | dict) -> str | None:
    """Return the data-provider ticker for a watchlist row.

    Falls back to the display ticker when ``data_ticker`` is absent or blank.

    Returns:
        The data-provider ticker string, or ``None`` when there is no usable
        ticker at all (both ``data_ticker`` and ``ticker`` are absent/blank).
    """
    raw = row.get("data_ticker")
    if raw is not None and str(raw).strip():
        return str(raw).strip()

    # Fall back to display ticker
    ticker = row.get("ticker")
    if ticker is not None and str(ticker).strip():
        return str(ticker).strip()

    return None


def has_valid_data_ticker(row: pd.Series | dict) -> bool:
    """Return True when the row has a non-empty data-provider ticker."""
    return get_data_ticker(row) is not None
