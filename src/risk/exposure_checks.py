import pandas as pd


def check_target_weight_sum(watchlist: pd.DataFrame) -> dict[str, float | bool]:
    total = float(watchlist["target_weight"].sum())
    return {
        "ok": abs(total - 100.0) < 1e-6,
        "total_target_weight": total,
    }
