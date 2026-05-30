from pathlib import Path

import pandas as pd

from src.data.asset_classifier import is_tradeable_asset

REQUIRED_COLUMNS = [
    "ticker",
    "name",
    "market",
    "type",
    "bucket",
    "currency",
    "target_weight",
]


def load_watchlist(path: str | Path) -> pd.DataFrame:
    """Load and validate watchlist CSV."""
    watchlist = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in watchlist.columns]
    if missing:
        raise ValueError(f"Missing watchlist columns: {missing}")

    watchlist["ticker"] = watchlist["ticker"].astype(str).str.upper().str.strip()
    watchlist["target_weight"] = pd.to_numeric(watchlist["target_weight"], errors="raise")
    watchlist["is_tradeable"] = watchlist.apply(is_tradeable_asset, axis=1)

    # Normalise optional data_ticker column; default to display ticker when absent.
    if "data_ticker" not in watchlist.columns:
        watchlist["data_ticker"] = watchlist["ticker"]
    else:
        watchlist["data_ticker"] = (
            watchlist["data_ticker"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        # Where data_ticker is blank, fall back to the display ticker.
        mask_empty = watchlist["data_ticker"] == ""
        watchlist.loc[mask_empty, "data_ticker"] = watchlist.loc[mask_empty, "ticker"]

    return watchlist
