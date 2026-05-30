from __future__ import annotations

import pandas as pd


def is_tradeable_asset(row: pd.Series | dict) -> bool:
    """Classify whether a watchlist row is a tradeable market instrument."""
    asset_type = str(row.get("type", "")).strip().upper()
    market = str(row.get("market", "")).strip().upper()
    return asset_type != "CASH" and market != "CASH"
