import numpy as np
import pandas as pd

from src.signals.signal_engine import evaluate_signal


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
