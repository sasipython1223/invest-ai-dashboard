import pandas as pd

from src.ui.charts import (
    build_allocation_by_bucket,
    build_entry_zone_dataframe,
    build_signal_distribution,
    calculate_action_urgency_score,
    calculate_watchlist_health_score,
)


def test_watchlist_health_excludes_cash_ticker():
    signals = pd.DataFrame(
        [
            {"ticker": "VWRA", "signal": "Hold / Buy Candidate"},
            {"ticker": "DBS", "signal": "Watch"},
            {"ticker": "CASH", "signal": "Hold / Buy Candidate"},
        ]
    )

    result = calculate_watchlist_health_score(signals)

    assert result["active_tradeable_count"] == 2
    assert result["hold_buy_count"] == 1
    assert result["score"] == 50


def test_watchlist_health_is_100_when_all_active_are_hold_buy():
    signals = pd.DataFrame(
        [
            {"ticker": "VWRA", "signal": "Hold / Buy Candidate"},
            {"ticker": "CSPX", "signal": "Hold / Buy Candidate"},
            {"ticker": "QQQ", "signal": "Insufficient Data"},
            {"ticker": "CASH", "signal": "Cash Reserve / No Signal"},
        ]
    )

    result = calculate_watchlist_health_score(signals)

    assert result["score"] == 100
    assert result["label"] == "Risk-On / Constructive"


def test_action_urgency_increases_with_reduce_avoid():
    watch_only = pd.DataFrame([{"signal": "Watch"}])
    with_reduce = pd.DataFrame(
        [
            {"signal": "Watch"},
            {"signal": "Reduce / Avoid"},
        ]
    )

    watch_score = calculate_action_urgency_score(watch_only)
    reduce_score = calculate_action_urgency_score(with_reduce)

    assert watch_score["score"] < reduce_score["score"]
    assert reduce_score["label"] == "Review"


def test_signal_distribution_counts_expected_labels():
    signals = pd.DataFrame(
        [
            {"signal": "Hold / Buy Candidate"},
            {"signal": "Hold / Buy Candidate"},
            {"signal": "Watch"},
            {"signal": "Reduce / Avoid"},
            {"signal": "Cash Reserve / No Signal"},
            {"signal": "Insufficient Data"},
        ]
    )

    distribution = build_signal_distribution(signals).set_index("signal_group")["count"].to_dict()

    assert distribution == {
        "Hold / Buy Candidate": 2,
        "Watch": 1,
        "Reduce / Avoid": 1,
        "Cash Reserve / No Signal": 1,
        "Insufficient Data": 1,
    }


def test_allocation_by_bucket_sums_target_weights():
    watchlist = pd.DataFrame(
        [
            {"bucket": "core", "target_weight": 40},
            {"bucket": "core", "target_weight": 20},
            {"bucket": "tactical", "target_weight": 3},
            {"bucket": "cash", "target_weight": 12},
        ]
    )

    allocation = build_allocation_by_bucket(watchlist).set_index("bucket")["target_weight"].to_dict()

    assert allocation["core"] == 60.0
    assert allocation["cash"] == 12.0
    assert allocation["tactical"] == 3.0


def test_entry_zone_dataframe_normalizes_inverted_bounds_for_display_only():
    entry_guidance = {
        "aggressive_bid_zone_bounds": {"low": 101.0, "high": 99.0},
        "normal_bid_zone_bounds": {"low": 98.0, "high": 100.0},
        "conservative_bid_zone_bounds": {"low": 95.0, "high": 94.0},
    }

    entry_zone_df = build_entry_zone_dataframe(entry_guidance).set_index("label")

    assert entry_zone_df.loc["Aggressive Bid Zone", "display_low"] == 99.0
    assert entry_zone_df.loc["Aggressive Bid Zone", "display_high"] == 101.0
    assert bool(entry_zone_df.loc["Aggressive Bid Zone", "was_normalized"]) is True
    assert bool(entry_zone_df.loc["Normal Bid Zone", "was_normalized"]) is False
    assert entry_guidance["aggressive_bid_zone_bounds"]["low"] == 101.0
    assert entry_guidance["aggressive_bid_zone_bounds"]["high"] == 99.0


def test_chart_helpers_do_not_require_external_api_data():
    empty_signals = pd.DataFrame(columns=["ticker", "signal"])
    empty_watchlist = pd.DataFrame(columns=["bucket", "target_weight"])

    assert calculate_watchlist_health_score(empty_signals)["score"] == 0
    assert calculate_action_urgency_score(empty_signals)["score"] == 0
    assert build_signal_distribution(empty_signals)["count"].sum() == 0
    assert build_allocation_by_bucket(empty_watchlist).empty
    assert build_entry_zone_dataframe({}).empty
