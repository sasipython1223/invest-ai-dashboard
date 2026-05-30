import pandas as pd
import pytest

from src.ui.trend_charts import (
    build_drawdown_series,
    build_indexed_price_series,
    build_return_summary,
    build_ticker_trend_dataframe,
    build_weighted_portfolio_index,
    rebase_comparison_frame,
    resolve_price_history,
)


def test_indexed_price_series_starts_at_100():
    history = pd.Series([10.0, 11.0, 12.0])

    indexed = build_indexed_price_series(history)

    assert indexed.iloc[0] == 100.0


def test_indexed_price_series_handles_empty_input_safely():
    indexed = build_indexed_price_series(pd.Series(dtype=float))

    assert indexed.empty


def test_weighted_portfolio_index_excludes_cash_and_normalizes_weights():
    watchlist = pd.DataFrame(
        [
            {"ticker": "AAA", "target_weight": 60},
            {"ticker": "BBB", "target_weight": 20},
            {"ticker": "CASH", "target_weight": 20},
        ]
    )
    prices = {
        "AAA": pd.Series([10.0, 11.0]),
        "BBB": pd.Series([100.0, 120.0]),
        "CASH": pd.Series([1.0, 10.0]),
    }

    weighted_index = build_weighted_portfolio_index(prices, watchlist)

    assert weighted_index.iloc[0] == 100.0
    assert round(float(weighted_index.iloc[-1]), 2) == 112.50


def test_drawdown_is_zero_at_new_highs_and_negative_below_peak():
    index_series = pd.Series([100.0, 110.0, 105.0, 120.0])

    drawdown = build_drawdown_series(index_series)

    assert drawdown.iloc[0] == 0.0
    assert drawdown.iloc[1] == 0.0
    assert drawdown.iloc[2] < 0.0
    assert drawdown.iloc[3] == 0.0


def test_rebase_comparison_frame_uses_first_shared_date_as_base_100():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    compare_df = pd.DataFrame(
        {
            "Portfolio/Watchlist": [110.0, 121.0],
            "VWRA": [100.0, 105.0],
        },
        index=dates,
    )

    rebased = rebase_comparison_frame(compare_df)

    assert rebased.iloc[0]["Portfolio/Watchlist"] == 100.0
    assert rebased.iloc[0]["VWRA"] == 100.0
    assert rebased.iloc[1]["Portfolio/Watchlist"] == pytest.approx(110.0)
    assert rebased.iloc[1]["VWRA"] == pytest.approx(105.0)


def test_ticker_trend_dataframe_includes_required_columns():
    history = pd.Series(range(1, 251), dtype=float)

    trend_df = build_ticker_trend_dataframe(history)

    assert {"price", "sma_20", "sma_50", "sma_200"}.issubset(trend_df.columns)


def test_return_summary_includes_3m_and_6m_momentum():
    signals = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "name": "Alpha",
                "bucket": "core",
                "signal": "Watch",
                "momentum_3m": 0.03,
                "momentum_6m": 0.08,
            }
        ]
    )
    prices = {"AAA": pd.Series(range(1, 40), dtype=float)}

    summary = build_return_summary(signals, prices)

    assert "momentum_3m" in summary.columns
    assert "momentum_6m" in summary.columns
    assert summary.loc[0, "momentum_3m"] == 0.03
    assert summary.loc[0, "momentum_6m"] == 0.08


def test_trend_helpers_do_not_make_external_api_calls(monkeypatch):
    from src.data import price_loader

    def _fail_external_call(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("External API should not be called by trend helpers.")

    monkeypatch.setattr(price_loader, "load_price_history", _fail_external_call)

    history = pd.Series([10.0, 11.0, 12.0, 13.0], dtype=float)
    signals = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "name": "Alpha",
                "bucket": "core",
                "signal": "Watch",
                "momentum_3m": 0.01,
                "momentum_6m": 0.02,
            }
        ]
    )
    prices = {"AAA": history}
    watchlist = pd.DataFrame([{"ticker": "AAA", "target_weight": 100}])

    assert not build_indexed_price_series(history).empty
    assert not build_weighted_portfolio_index(prices, watchlist).empty
    assert not build_drawdown_series(pd.Series([100.0, 95.0])).empty
    assert not build_ticker_trend_dataframe(history).empty
    assert not build_return_summary(signals, prices).empty


# ---------------------------------------------------------------------------
# resolve_price_history tests
# ---------------------------------------------------------------------------


def test_resolve_price_history_finds_display_ticker_key():
    prices = {"VWRA": pd.Series([100.0, 110.0])}
    result = resolve_price_history(prices, "VWRA", "VWRA.L")
    assert not result.empty
    assert list(result) == [100.0, 110.0]


def test_resolve_price_history_falls_back_to_data_ticker():
    prices = {"VWRA.L": pd.Series([100.0, 110.0])}
    result = resolve_price_history(prices, "VWRA", "VWRA.L")
    assert not result.empty
    assert list(result) == [100.0, 110.0]


def test_resolve_price_history_returns_empty_when_neither_key_exists():
    prices = {"OTHER": pd.Series([100.0, 110.0])}
    result = resolve_price_history(prices, "VWRA", "VWRA.L")
    assert result.empty


def test_resolve_price_history_returns_empty_series_without_api_calls(monkeypatch):
    from src.data import price_loader

    def _fail(*args, **kwargs):  # pragma: no cover - defensive
        raise AssertionError("External API must not be called.")

    monkeypatch.setattr(price_loader, "load_price_history", _fail)
    result = resolve_price_history({}, "VWRA", "VWRA.L")
    assert result.empty


def test_resolve_price_history_no_data_ticker_fallback():
    prices = {"VWRA.L": pd.Series([50.0, 60.0])}
    # No data_ticker provided — should return empty
    result = resolve_price_history(prices, "VWRA")
    assert result.empty


# ---------------------------------------------------------------------------
# build_weighted_portfolio_index with data_ticker mapping
# ---------------------------------------------------------------------------


def test_weighted_portfolio_index_works_when_prices_keyed_by_data_ticker():
    watchlist = pd.DataFrame(
        [
            {"ticker": "VWRA", "data_ticker": "VWRA.L", "target_weight": 60},
            {"ticker": "ES3", "data_ticker": "ES3.SI", "target_weight": 40},
        ]
    )
    # Prices keyed by data_ticker only
    prices = {
        "VWRA.L": pd.Series([100.0, 110.0]),
        "ES3.SI": pd.Series([200.0, 220.0]),
    }

    result = build_weighted_portfolio_index(prices, watchlist)

    assert not result.empty
    assert result.iloc[0] == 100.0


def test_weighted_portfolio_index_excludes_cash_with_data_ticker_column():
    watchlist = pd.DataFrame(
        [
            {"ticker": "AAA", "data_ticker": "AAA.L", "target_weight": 80},
            {"ticker": "CASH", "data_ticker": None, "target_weight": 20},
        ]
    )
    prices = {"AAA.L": pd.Series([10.0, 11.0, 12.0])}

    result = build_weighted_portfolio_index(prices, watchlist)

    assert not result.empty
    assert result.iloc[0] == 100.0


# ---------------------------------------------------------------------------
# Benchmark lookup with data_ticker mapping (helper extracted for test)
# ---------------------------------------------------------------------------


def _resolve_benchmark(prices: dict, watchlist: pd.DataFrame):
    """Mirror of the dashboard benchmark resolution logic for testing."""
    from src.ui.trend_charts import build_indexed_price_series, resolve_price_history

    ticker_to_data_ticker: dict[str, str] = {}
    if "data_ticker" in watchlist.columns:
        for _, row in watchlist.iterrows():
            t = str(row.get("ticker", "")).strip()
            dt = str(row.get("data_ticker", "") or "").strip()
            if t and dt:
                ticker_to_data_ticker[t] = dt

    for candidate in ["VWRA", "CSPX", "ES3"]:
        dt = ticker_to_data_ticker.get(candidate)
        idx = build_indexed_price_series(resolve_price_history(prices, candidate, dt))
        if not idx.empty:
            return candidate, idx
    return None, pd.Series(dtype=float)


def test_benchmark_lookup_resolves_via_data_ticker():
    watchlist = pd.DataFrame(
        [
            {"ticker": "VWRA", "data_ticker": "VWRA.L"},
            {"ticker": "CSPX", "data_ticker": "CSPX.L"},
            {"ticker": "ES3", "data_ticker": "ES3.SI"},
        ]
    )
    prices = {"VWRA.L": pd.Series([100.0, 105.0, 110.0])}

    label, series = _resolve_benchmark(prices, watchlist)

    assert label == "VWRA"
    assert not series.empty


def test_benchmark_lookup_falls_through_to_next_candidate():
    watchlist = pd.DataFrame(
        [
            {"ticker": "VWRA", "data_ticker": "VWRA.L"},
            {"ticker": "CSPX", "data_ticker": "CSPX.L"},
            {"ticker": "ES3", "data_ticker": "ES3.SI"},
        ]
    )
    # Only CSPX available
    prices = {"CSPX.L": pd.Series([400.0, 420.0, 450.0])}

    label, series = _resolve_benchmark(prices, watchlist)

    assert label == "CSPX"
    assert not series.empty
