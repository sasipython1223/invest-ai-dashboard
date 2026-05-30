from __future__ import annotations

import pandas as pd


def get_data_ticker(row: pd.Series | dict) -> str | None:
    """Return the data-provider ticker for a watchlist row.

    Falls back to the display ticker when ``data_ticker`` is absent or blank.
    Returns ``None`` only when there is no usable ticker at all.
    """
    raw = row.get("data_ticker") if isinstance(row, dict) else row.get("data_ticker")
    if raw is not None and str(raw).strip():
        return str(raw).strip()

    # Fall back to display ticker
    ticker = row.get("ticker") if isinstance(row, dict) else row.get("ticker")
    if ticker is not None and str(ticker).strip():
        return str(ticker).strip()

    return None


def has_valid_data_ticker(row: pd.Series | dict) -> bool:
    """Return True when the row has a non-empty data-provider ticker."""
    return get_data_ticker(row) is not None
