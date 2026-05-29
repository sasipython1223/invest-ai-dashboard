from pathlib import Path

import pandas as pd

COLUMNS = [
    "timestamp",
    "ticker",
    "action",
    "rationale",
    "risk_status",
    "manual_checklist_completed",
]


def load_trade_journal(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)
