from __future__ import annotations

import pandas as pd


def format_optional_price(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.3f}"


def build_bid_zone_table(entry_guidance: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Entry style": "Aggressive",
                "Bid zone": entry_guidance.get("aggressive_bid_zone") or "N/A",
                "Interpretation": "Higher fill chance, lower margin of safety",
            },
            {
                "Entry style": "Normal",
                "Bid zone": entry_guidance.get("normal_bid_zone") or "N/A",
                "Interpretation": "Balanced review zone",
            },
            {
                "Entry style": "Conservative",
                "Bid zone": entry_guidance.get("conservative_bid_zone") or "N/A",
                "Interpretation": "More safety margin, lower fill chance",
            },
        ]
    )
