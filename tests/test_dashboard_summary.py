import pandas as pd

from src.dashboard_summary import MANUAL_REVIEW_NOTE, build_action_items, count_signals


def test_count_signals_counts_active_signal_groups_only():
    signals = pd.DataFrame(
        [
            {"ticker": "VWRA", "signal": "Hold / Buy Candidate"},
            {"ticker": "CSPX", "signal": "Hold / Buy Candidate"},
            {"ticker": "DBS", "signal": "Watch"},
            {"ticker": "UOB", "signal": "Reduce / Avoid"},
            {"ticker": "CASH", "signal": "Cash Reserve / No Signal"},
            {"ticker": "QQQ", "signal": "Insufficient Data"},
        ]
    )

    result = count_signals(signals)

    assert result == {
        "total_instruments": 6,
        "hold_buy_candidates": 2,
        "watch_items": 1,
        "reduce_avoid_items": 1,
    }



def test_build_action_items_orders_active_groups_and_includes_manual_review_note():
    signals = pd.DataFrame(
        [
            {"ticker": "VWRA", "signal": "Hold / Buy Candidate"},
            {"ticker": "CSPX", "signal": "Hold / Buy Candidate"},
            {"ticker": "DBS", "signal": "Watch"},
            {"ticker": "UOB", "signal": "Reduce / Avoid"},
            {"ticker": "CASH", "signal": "Cash Reserve / No Signal"},
        ]
    )

    result = build_action_items(signals, {"reserve_target_weight": 12.0})

    assert result[0] == f"1. Review Reduce / Avoid items: UOB. {MANUAL_REVIEW_NOTE}"
    assert result[1] == f"2. Watch items: DBS. {MANUAL_REVIEW_NOTE}"
    assert result[2] == f"3. Hold / Buy candidates: VWRA, CSPX. {MANUAL_REVIEW_NOTE}"
    assert result[3] == "4. Cash reserve: 12.00% target weight."



def test_build_action_items_uses_no_action_state_when_no_active_signals():
    signals = pd.DataFrame(
        [
            {"ticker": "CASH", "signal": "Cash Reserve / No Signal"},
            {"ticker": "QQQ", "signal": "Insufficient Data"},
        ]
    )

    result = build_action_items(signals, {"reserve_target_weight": 10.0})

    assert result == [
        "1. No urgent signal changes today.",
        "2. Cash reserve: 10.00% target weight.",
    ]
