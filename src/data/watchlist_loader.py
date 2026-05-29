from pathlib import Path

import pandas as pd

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
    return watchlist
