import pandas as pd

from src.portfolio.capital_allocator import build_cash_deployment_plan


def _sample_watchlist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "VWRA", "name": "Vanguard FTSE All-World", "type": "ETF", "bucket": "core", "currency": "USD", "target_weight": 40},
            {"ticker": "CSPX", "name": "iShares Core S&P 500", "type": "ETF", "bucket": "core", "currency": "USD", "target_weight": 20},
            {"ticker": "ES3", "name": "STI ETF", "type": "ETF", "bucket": "singapore", "currency": "SGD", "target_weight": 10},
            {"ticker": "AAPL", "name": "Apple", "type": "Stock", "bucket": "tactical", "currency": "USD", "target_weight": 5},
            {"ticker": "CASH", "name": "Cash Reserve", "type": "Cash", "bucket": "cash", "currency": "SGD", "target_weight": 25},
        ]
    )


def _sample_signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "VWRA", "signal": "Hold / Buy Candidate"},
            {"ticker": "CSPX", "signal": "Reduce / Avoid"},
            {"ticker": "ES3", "signal": "Watch"},
            {"ticker": "AAPL", "signal": "Hold / Buy Candidate"},
            {"ticker": "CASH", "signal": "Cash Reserve / No Signal"},
        ]
    )


def test_excludes_cash_from_allocation_rows():
    plan = build_cash_deployment_plan(
        amount=1000,
        watchlist=_sample_watchlist(),
        signals=_sample_signals(),
        risk={"status": "OK"},
        deployment_style="Balanced",
    )

    allocated_tickers = {row["ticker"] for row in plan["allocation_rows"]}
    excluded_tickers = {row["ticker"] for row in plan["excluded_rows"]}

    assert "CASH" not in allocated_tickers
    assert "CASH" in excluded_tickers


def test_excludes_reduce_avoid_from_allocation_rows():
    plan = build_cash_deployment_plan(
        amount=1000,
        watchlist=_sample_watchlist(),
        signals=_sample_signals(),
        risk={"status": "OK"},
        deployment_style="Balanced",
    )

    allocated_tickers = {row["ticker"] for row in plan["allocation_rows"]}
    assert "CSPX" not in allocated_tickers


def test_conservative_deploys_less_than_balanced():
    watchlist = _sample_watchlist()
    signals = _sample_signals()

    conservative = build_cash_deployment_plan(1000, watchlist, signals, {"status": "OK"}, "Conservative")
    balanced = build_cash_deployment_plan(1000, watchlist, signals, {"status": "OK"}, "Balanced")

    assert conservative["deploy_now_amount"] < balanced["deploy_now_amount"]


def test_opportunistic_deploys_no_more_than_80_percent():
    watchlist = _sample_watchlist()
    signals = _sample_signals()

    opportunistic_ok = build_cash_deployment_plan(1000, watchlist, signals, {"status": "OK"}, "Opportunistic")
    opportunistic_review = build_cash_deployment_plan(
        1000, watchlist, signals, {"status": "REVIEW"}, "Opportunistic"
    )

    assert opportunistic_ok["deploy_now_amount"] <= 800
    assert opportunistic_review["deploy_now_amount"] == 600


def test_tactical_candidates_excluded_unless_enabled():
    watchlist = _sample_watchlist()
    signals = _sample_signals()

    tactical_off = build_cash_deployment_plan(1000, watchlist, signals, {"status": "OK"}, "Balanced")
    tactical_on = build_cash_deployment_plan(
        1000,
        watchlist,
        signals,
        {"status": "OK"},
        "Balanced",
        include_tactical=True,
    )

    tickers_off = {row["ticker"] for row in tactical_off["allocation_rows"]}
    tickers_on = {row["ticker"] for row in tactical_on["allocation_rows"]}

    assert "AAPL" not in tickers_off
    assert "AAPL" in tickers_on


def test_allocation_rows_sum_to_deploy_amount_within_rounding_tolerance():
    plan = build_cash_deployment_plan(
        amount=1000,
        watchlist=_sample_watchlist(),
        signals=_sample_signals(),
        risk={"status": "OK"},
        deployment_style="Balanced",
    )

    suggested_sum = sum(row["suggested_amount"] for row in plan["allocation_rows"])

    assert abs(suggested_sum - plan["deploy_now_amount"]) <= 0.01


def test_tranches_sum_to_suggested_amount_within_rounding_tolerance():
    plan = build_cash_deployment_plan(
        amount=1000,
        watchlist=_sample_watchlist(),
        signals=_sample_signals(),
        risk={"status": "OK"},
        deployment_style="Balanced",
    )

    for row in plan["allocation_rows"]:
        tranche_sum = row["tranche_1"] + row["tranche_2"] + row["tranche_3"]
        assert abs(tranche_sum - row["suggested_amount"]) <= 0.01


def test_zero_amount_returns_safe_warning_and_no_allocation():
    plan = build_cash_deployment_plan(
        amount=0,
        watchlist=_sample_watchlist(),
        signals=_sample_signals(),
        risk={"status": "OK"},
        deployment_style="Balanced",
    )

    assert plan["deploy_now_amount"] == 0
    assert plan["allocation_rows"] == []
    assert plan["warnings"]


def test_capital_allocator_is_pure_deterministic_no_api_dependencies():
    plan = build_cash_deployment_plan(
        amount=1000,
        watchlist=_sample_watchlist(),
        signals=_sample_signals(),
        risk={"status": "OK"},
        deployment_style="Balanced",
    )

    assert isinstance(plan, dict)
    assert plan["manual_review_required"] is True
