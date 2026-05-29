import pandas as pd

from src.risk.risk_engine import evaluate_risk


def test_risk_engine_ok_status():
    watchlist = pd.DataFrame(
        [
            {"ticker": "AAA", "target_weight": 60},
            {"ticker": "BBB", "target_weight": 40},
        ]
    )
    signals = pd.DataFrame([{"signal": "Hold / Buy Candidate"}])

    result = evaluate_risk(watchlist, signals)

    assert result["status"] == "OK"


def test_risk_engine_review_when_weights_invalid_or_reduce_signal():
    watchlist = pd.DataFrame(
        [
            {"ticker": "AAA", "target_weight": 70},
            {"ticker": "BBB", "target_weight": 40},
        ]
    )
    signals = pd.DataFrame([{"signal": "Reduce / Avoid"}])

    result = evaluate_risk(watchlist, signals)

    assert result["status"] == "REVIEW"
    assert len(result["alerts"]) >= 2


def test_risk_engine_excludes_cash_from_reduce_avoid_count():
    watchlist = pd.DataFrame(
        [
            {"ticker": "AAA", "market": "NASDAQ", "type": "Stock", "target_weight": 90},
            {"ticker": "CASH", "market": "CASH", "type": "Cash", "target_weight": 10},
        ]
    )
    signals = pd.DataFrame(
        [
            {"ticker": "AAA", "signal": "Reduce / Avoid"},
            {"ticker": "CASH", "signal": "Reduce / Avoid"},
        ]
    )

    result = evaluate_risk(watchlist, signals)

    assert result["status"] == "REVIEW"
    assert "1 holding flagged as Reduce / Avoid." in result["alerts"]
    assert result["reserve_target_weight"] == 10.0
