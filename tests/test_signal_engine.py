import numpy as np
import pandas as pd

from src.signals.signal_engine import (
    SIGNAL_CASH_REASON,
    SIGNAL_CASH_RESERVE,
    evaluate_signal,
    run_signal_engine,
)


def _series(values):
    return pd.Series(np.array(values, dtype=float))


def test_signal_hold_buy_candidate():
    down = np.linspace(100, 95, 100)
    up = np.linspace(95, 130, 120)
    series = _series(np.concatenate([down, up]))

    result = evaluate_signal(series)

    assert result["signal"] == "Hold / Buy Candidate"


def test_signal_watch_when_short_term_weakness():
    older_low_regime = np.full(80, 50)
    recent_decline = np.linspace(160, 120, 140)
    series = _series(np.concatenate([older_low_regime, recent_decline]))

    result = evaluate_signal(series)

    assert result["signal"] == "Watch"


def test_signal_reduce_avoid():
    series = _series(np.linspace(200, 80, 220))

    result = evaluate_signal(series)

    assert result["signal"] == "Reduce / Avoid"


def test_signal_insufficient_data():
    series = _series(np.linspace(100, 120, 40))

    result = evaluate_signal(series)

    assert result["signal"] == "Insufficient Data"


def test_run_signal_engine_cash_reserve_has_no_technical_signal():
    watchlist = pd.DataFrame(
        [
            {"ticker": "CASH", "name": "Cash Reserve", "market": "CASH", "type": "Cash", "bucket": "cash"},
            {"ticker": "AAPL", "name": "Apple", "market": "NASDAQ", "type": "Stock", "bucket": "tactical"},
        ]
    )
    prices = {"AAPL": _series(np.linspace(100, 130, 220))}

    signals = run_signal_engine(watchlist, prices).set_index("ticker")

    assert signals.loc["CASH", "signal"] == SIGNAL_CASH_RESERVE
    assert signals.loc["CASH", "signal_reason"] == SIGNAL_CASH_REASON
    assert pd.isna(signals.loc["CASH", "latest_price"])
    assert signals.loc["AAPL", "signal"] != SIGNAL_CASH_RESERVE
