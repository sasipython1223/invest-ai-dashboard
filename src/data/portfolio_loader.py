from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["ticker", "quantity", "average_cost", "currency", "bucket"]


def load_portfolio(path: str | Path) -> pd.DataFrame:
    """Load and validate portfolio CSV."""
    portfolio = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in portfolio.columns]
    if missing:
        raise ValueError(f"Missing portfolio columns: {missing}")

    portfolio["ticker"] = portfolio["ticker"].astype(str).str.upper().str.strip()
    return portfolio
