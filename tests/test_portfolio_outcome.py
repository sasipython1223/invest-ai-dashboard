import inspect

import pandas as pd

from src.analytics import portfolio_outcome
from src.analytics.portfolio_outcome import (
    build_default_allocation,
    calculate_portfolio_outcome,
    calculate_ticker_outcome,
    normalize_allocations,
)


def _sample_watchlist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "VWRA", "name": "Vanguard FTSE All-World", "type": "ETF", "bucket": "core", "target_weight": 50},
            {"ticker": "CSPX", "name": "iShares Core S&P 500", "type": "ETF", "bucket": "core", "target_weight": 25},
            {"ticker": "ES3", "name": "SPDR STI ETF", "type": "ETF", "bucket": "singapore", "target_weight": 12.5},
            {"ticker": "DBS", "name": "DBS Group", "type": "Stock", "bucket": "singapore", "target_weight": 6.25},
            {"ticker": "UOB", "name": "UOB", "type": "Stock", "bucket": "singapore", "target_weight": 6.25},
            {"ticker": "AAPL", "name": "Apple", "type": "Stock", "bucket": "tactical", "target_weight": 5},
            {"ticker": "CASH", "name": "Cash Reserve", "type": "Cash", "bucket": "cash", "target_weight": 10},
        ]
    )


def _sample_signals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ticker": "VWRA", "signal": "Hold / Buy Candidate"},
            {"ticker": "CSPX", "signal": "Hold / Buy Candidate"},
            {"ticker": "ES3", "signal": "Hold / Buy Candidate"},
            {"ticker": "DBS", "signal": "Hold / Buy Candidate"},
            {"ticker": "UOB", "signal": "Hold / Buy Candidate"},
            {"ticker": "AAPL", "signal": "Hold / Buy Candidate"},
            {"ticker": "CASH", "signal": "Cash Reserve / No Signal"},
        ]
    )


def test_default_allocation_excludes_cash_and_reduce_avoid():
    signals = _sample_signals()
    signals.loc[signals["ticker"] == "DBS", "signal"] = "Reduce / Avoid"

    allocation = build_default_allocation(signals=signals, watchlist=_sample_watchlist())
    tickers = set(allocation["ticker"])

    assert "CASH" not in tickers
    assert "DBS" not in tickers


def test_default_allocation_excludes_tactical_when_disabled():
    allocation = build_default_allocation(
        signals=_sample_signals(),
        watchlist=_sample_watchlist(),
        include_tactical=False,
    )

    assert "AAPL" not in set(allocation["ticker"])


def test_default_allocation_includes_tactical_when_enabled():
    allocation = build_default_allocation(
        signals=_sample_signals(),
        watchlist=_sample_watchlist(),
        include_tactical=True,
    )

    assert "AAPL" in set(allocation["ticker"])


def test_default_allocation_normalizes_target_weights_to_100_percent():
    allocation = build_default_allocation(
        signals=_sample_signals(),
        watchlist=_sample_watchlist(),
    )

    assert abs(float(allocation["allocation_pct"].sum()) - 100.0) < 1e-9


def test_normalize_allocations_handles_total_not_100():
    allocation = pd.DataFrame(
        [
            {"ticker": "VWRA", "allocation_pct": 70.0},
            {"ticker": "CSPX", "allocation_pct": 20.0},
        ]
    )

    normalized = normalize_allocations(allocation)

    assert abs(float(normalized["allocation_pct"].sum()) - 100.0) < 1e-9
    assert float(normalized.loc[normalized["ticker"] == "VWRA", "allocation_pct"].iloc[0]) > 70.0


def test_ticker_outcome_uses_scenario_ratio_to_convert_to_currency(monkeypatch):
    history = pd.Series([100.0, 110.0, 120.0])

    def _fake_summary(_: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "horizon": "1M",
                    "best_2sd": 132.0,
                    "normal_upper_1sd": 126.0,
                    "expected": 126.0,
                    "normal_lower_1sd": 120.0,
                    "worst_2sd": 114.0,
                }
            ]
        )

    monkeypatch.setattr(portfolio_outcome, "build_scenario_summary", _fake_summary)
    outcome = calculate_ticker_outcome("VWRA", 200.0, history, "1M")

    assert outcome["status"] == "ok"
    assert abs(float(outcome["expected_value"]) - 210.0) < 1e-9
    assert abs(float(outcome["worst_value"]) - 190.0) < 1e-9
    assert abs(float(outcome["expected_gain_loss"]) - 10.0) < 1e-9
    assert abs(float(outcome["worst_loss"]) + 10.0) < 1e-9


def test_portfolio_outcome_aggregates_ticker_outcomes(monkeypatch):
    prices = {
        "VWRA": pd.Series([100.0, 110.0, 120.0]),
        "CSPX": pd.Series([50.0, 55.0, 60.0]),
    }
    allocation_df = pd.DataFrame(
        [
            {"ticker": "VWRA", "allocation_pct": 60.0},
            {"ticker": "CSPX", "allocation_pct": 40.0},
        ]
    )

    def _fake_summary(series: pd.Series) -> pd.DataFrame:
        current = float(series.iloc[-1])
        return pd.DataFrame(
            [
                {
                    "horizon": "1M",
                    "best_2sd": current * 1.20,
                    "normal_upper_1sd": current * 1.10,
                    "expected": current * 1.05,
                    "normal_lower_1sd": current * 0.95,
                    "worst_2sd": current * 0.90,
                }
            ]
        )

    monkeypatch.setattr(portfolio_outcome, "build_scenario_summary", _fake_summary)
    ticker_df, portfolio_df = calculate_portfolio_outcome(
        allocation_df=allocation_df,
        prices=prices,
        total_investment=1000.0,
        horizon_label="1M",
    )

    likely_row = portfolio_df.loc[portfolio_df["Scenario"] == "Likely / Expected"].iloc[0]
    worst_row = portfolio_df.loc[portfolio_df["Scenario"] == "Worst (-2 SD)"].iloc[0]

    assert len(ticker_df) == 2
    assert abs(float(likely_row["Ending Value"]) - 1050.0) < 1e-9
    assert abs(float(worst_row["Ending Value"]) - 900.0) < 1e-9


def test_insufficient_price_history_skips_ticker_with_clear_status():
    outcome = calculate_ticker_outcome(
        ticker="VWRA",
        allocated_amount=100.0,
        price_history=pd.Series([100.0]),
        horizon_label="1M",
    )

    assert outcome["status"] == "skipped_insufficient_history"


def test_portfolio_outcome_module_has_no_broker_or_ai_execution_logic():
    module_source = inspect.getsource(portfolio_outcome).lower()

    assert "broker" not in module_source
    assert "tiger api" not in module_source
    assert "order placement" not in module_source
    assert "review_with_openai" not in module_source
    assert "review_with_gemini" not in module_source
