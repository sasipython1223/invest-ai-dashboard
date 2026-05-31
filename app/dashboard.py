from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.analytics.scenario_forecast import (
    assign_volatility_risk_status,
    build_scenario_summary,
    build_volatility_cone,
)
from src.ai_review.consensus_checker import check_consensus
from src.ai_review.gemini_entry_reviewer import (
    DEFAULT_GEMINI_MODEL,
    build_gemini_entry_review_prompt,
    review_entry_with_gemini,
)
from src.ai_review.gemini_reviewer import review_with_gemini
from src.ai_review.openai_reviewer import review_with_openai
from src.data.portfolio_loader import load_portfolio
from src.data.price_loader import load_prices_for_watchlist
from src.dashboard_summary import MANUAL_REVIEW_NOTE, build_action_items, count_signals
from src.data.watchlist_loader import load_watchlist
from src.entries.entry_guidance import calculate_entry_guidance
from src.portfolio.capital_allocator import build_cash_deployment_plan
from src.risk.risk_engine import evaluate_risk
from src.signals.signal_engine import run_signal_engine
from src.ui.charts import (
    build_allocation_by_bucket,
    build_entry_zone_dataframe,
    build_signal_distribution,
    calculate_action_urgency_score,
    calculate_watchlist_health_score,
)
from src.ui.formatters import build_bid_zone_table, format_optional_price
from src.ui.tab_helpers import get_tab_labels
from src.ui.trend_charts import (
    build_drawdown_series,
    build_indexed_price_series,
    build_return_summary,
    build_ticker_trend_dataframe,
    build_weighted_portfolio_index,
    get_portfolio_index_diagnostics,
    get_price_history_diagnostics,
    rebase_comparison_frame,
    resolve_price_history,
)
from src.utils.config import load_config


st.set_page_config(page_title="invest-ai-dashboard", layout="wide")
st.title("invest-ai-dashboard")
st.caption("Rules decide. AI explains. User executes manually.")

GUARDRAIL_SUMMARY = (
    "Educational prototype only. Not financial advice. "
    "No auto-trading or broker API/Tiger API execution. "
    "All trades must be manually reviewed and executed by the user."
)
STATUS_NOT_ENABLED = "Not enabled"
STATUS_PLACEHOLDER = "Placeholder"
STATUS_INSUFFICIENT_AI = "Insufficient AI reviews"
BENCHMARK_CANDIDATES = ["VWRA", "CSPX", "ES3"]
SCENARIO_WARNINGS = (
    "Scenario range based on historical volatility. Not a prediction or guarantee.",
    "Models assume normal market conditions. Extreme events can exceed the displayed bands.",
    "Longer-horizon forecasts are less reliable than shorter-horizon forecasts.",
    "Manual review required. No trades are executed by this dashboard.",
)
st.warning(GUARDRAIL_SUMMARY)


def _build_scenario_cone_figure(cone_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cone_df["step"],
            y=cone_df["extreme_upper"],
            mode="lines",
            line=dict(width=0),
            name="Extreme Upper (+2 SD)",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cone_df["step"],
            y=cone_df["extreme_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 140, 0, 0.15)",
            name="Extreme band (±2 SD)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cone_df["step"],
            y=cone_df["normal_upper"],
            mode="lines",
            line=dict(width=0),
            name="Normal Upper (+1 SD)",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cone_df["step"],
            y=cone_df["normal_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(30, 144, 255, 0.22)",
            name="Normal band (±1 SD)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cone_df["step"],
            y=cone_df["expected"],
            mode="lines",
            line=dict(width=2),
            name="Expected / Median",
        )
    )
    fig.update_layout(
        title=title,
        height=340,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Trading days ahead",
        yaxis_title="Scenario value",
    )
    return fig

# ---------------------------------------------------------------------------
# Data loading — runs once per page load, shared across all tabs
# ---------------------------------------------------------------------------
config = load_config()
watchlist = load_watchlist(config["watchlist_path"])
portfolio = load_portfolio(config["portfolio_path"])
prices = load_prices_for_watchlist(watchlist)
signals = run_signal_engine(watchlist, prices)
risk = evaluate_risk(watchlist, signals)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tabs = st.tabs(get_tab_labels())

# ==========================================================================
# Tab 1 — Overview  (30-second control tower)
# ==========================================================================
with tabs[0]:
    st.header("Decision Center")
    st.subheader("What needs my decision today?")
    st.write("Deterministic signal output with manual execution checklist.")

    signal_counts = count_signals(signals)
    metric_columns = st.columns(5)
    metric_columns[0].metric("Total instruments", signal_counts["total_instruments"])
    metric_columns[1].metric("Hold / Buy candidates", signal_counts["hold_buy_candidates"])
    metric_columns[2].metric("Watch items", signal_counts["watch_items"])
    metric_columns[3].metric("Reduce / Avoid items", signal_counts["reduce_avoid_items"])
    metric_columns[4].metric("Cash reserve target weight", f"{risk.get('reserve_target_weight', 0.0):.2f}%")

    health_summary = calculate_watchlist_health_score(signals)
    urgency_summary = calculate_action_urgency_score(signals)
    gauge_cols = st.columns(2)

    with gauge_cols[0]:
        st.caption(f"Watchlist Health: {health_summary['label']}")
        st.plotly_chart(
            go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=health_summary["score"],
                    title={"text": "Watchlist Health"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "steps": [
                            {"range": [0, 39]},
                            {"range": [40, 69]},
                            {"range": [70, 100]},
                        ],
                    },
                )
            ).update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10)),
            use_container_width=True,
        )

    with gauge_cols[1]:
        st.caption(f"Action Urgency: {urgency_summary['label']}")
        st.plotly_chart(
            go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=urgency_summary["score"],
                    title={"text": "Action Urgency"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "steps": [
                            {"range": [0, 24]},
                            {"range": [25, 59]},
                            {"range": [60, 100]},
                        ],
                    },
                )
            ).update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10)),
            use_container_width=True,
        )

    reserve_target_weight = float(risk.get("reserve_target_weight", 0.0))
    st.caption(f"Cash reserve target: {reserve_target_weight:.2f}%")
    st.progress(max(0.0, min(1.0, reserve_target_weight / 100)))

    st.info(MANUAL_REVIEW_NOTE)
    st.subheader("Today's Action List")
    for action_item in build_action_items(signals, risk):
        st.write(action_item)

    st.subheader("Risk Alerts")
    st.write(f"Risk status: **{risk['status']}**")
    if risk.get("reserve_target_weight", 0.0) > 0:
        st.write(f"- Cash reserve (dry powder): {risk['reserve_target_weight']:.2f}% target weight.")
    for alert in risk["alerts"]:
        st.write(f"- {alert}")

    st.subheader("Signal Distribution")
    signal_distribution = build_signal_distribution(signals)
    st.plotly_chart(
        go.Figure(
            go.Pie(
                labels=signal_distribution["signal_group"],
                values=signal_distribution["count"],
                hole=0.55,
            )
        ).update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10)),
        use_container_width=True,
    )

    st.subheader("Target Allocation by Bucket")
    allocation_by_bucket = build_allocation_by_bucket(watchlist)
    if allocation_by_bucket.empty:
        st.info("Target allocation data is not available yet.")
    else:
        st.plotly_chart(
            go.Figure(
                go.Pie(
                    labels=allocation_by_bucket["bucket"],
                    values=allocation_by_bucket["target_weight"],
                    hole=0.55,
                )
            ).update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10)),
            use_container_width=True,
        )

# ==========================================================================
# Tab 2 — Portfolio  (capital management)
# ==========================================================================
with tabs[1]:
    st.header("Portfolio & Cash Deployment Review")
    st.caption(
        "This tab covers portfolio-level capital management. "
        "The watchlist (investment universe) is in the Watchlist tab."
    )

    st.subheader("Current portfolio snapshot")
    st.dataframe(portfolio, use_container_width=True)

    st.subheader("Combined indexed trend")
    # Build a data_ticker lookup from the watchlist for diagnostics and fallback resolution
    _ticker_to_data_ticker: dict[str, str] = {}
    if "data_ticker" in watchlist.columns:
        for _, _wl_row in watchlist.iterrows():
            _t = str(_wl_row.get("ticker", "")).strip()
            _dt = str(_wl_row.get("data_ticker", "") or "").strip()
            if _t and _dt:
                _ticker_to_data_ticker[_t] = _dt

    portfolio_index = build_weighted_portfolio_index(prices, watchlist)
    portfolio_index_diag = get_portfolio_index_diagnostics(prices, watchlist)

    if portfolio_index.empty:
        st.info("Combined indexed trend is unavailable due to missing or insufficient price history.")
        with st.expander("Show price history diagnostics"):
            st.write(f"Available price-history keys: **{len(prices)}**")
            st.write(f"Ticker → data_ticker mapping: {_ticker_to_data_ticker}")
            if not portfolio_index_diag.empty:
                st.write("Portfolio index diagnostics")
                st.dataframe(portfolio_index_diag, use_container_width=True)
            _diag_df = get_price_history_diagnostics(prices)
            if not _diag_df.empty:
                st.dataframe(_diag_df, use_container_width=True)
    else:
        portfolio_index_fig = go.Figure()
        portfolio_index_fig.add_trace(
            go.Scatter(
                x=portfolio_index.index,
                y=portfolio_index.values,
                mode="lines",
                name="Portfolio/Watchlist Index",
            )
        )
        portfolio_index_fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Date",
            yaxis_title="Index (Start = 100)",
        )
        st.plotly_chart(portfolio_index_fig, use_container_width=True)
        st.caption("Indexed trend based on available price history and target weights.")

    st.subheader("Portfolio vs benchmark")
    benchmark_series = pd.Series(dtype=float)
    benchmark_label = None
    for candidate in BENCHMARK_CANDIDATES:
        _candidate_dt = _ticker_to_data_ticker.get(candidate)
        candidate_index = build_indexed_price_series(
            resolve_price_history(prices, candidate, _candidate_dt)
        )
        if not candidate_index.empty:
            benchmark_label = candidate
            benchmark_series = candidate_index
            break

    if portfolio_index.empty:
        st.info("Portfolio index is unavailable, so benchmark comparison cannot be shown.")
    elif benchmark_label is None or benchmark_series.empty:
        st.info("Benchmark history is unavailable. Add VWRA, CSPX, or ES3 price history to enable comparison.")
        with st.expander("Show price history diagnostics"):
            st.write(f"Available price-history keys: **{len(prices)}**")
            st.write(f"Benchmark candidates checked: {BENCHMARK_CANDIDATES}")
            _benchmark_mapping = ", ".join(
                f"{c}→{_ticker_to_data_ticker.get(c, 'N/A')}" for c in BENCHMARK_CANDIDATES
            )
            st.write(f"Resolved data_ticker for benchmarks: {_benchmark_mapping}")
            _diag_df = get_price_history_diagnostics(prices)
            if not _diag_df.empty:
                st.dataframe(_diag_df, use_container_width=True)
    else:
        compare_df = pd.concat(
            [
                portfolio_index.rename("Portfolio/Watchlist"),
                benchmark_series.rename(benchmark_label),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if compare_df.empty:
            st.info("Portfolio and benchmark histories do not overlap enough for a comparison chart.")
        else:
            compare_df = rebase_comparison_frame(compare_df)
            comparison_fig = go.Figure()
            comparison_fig.add_trace(
                go.Scatter(
                    x=compare_df.index,
                    y=compare_df["Portfolio/Watchlist"],
                    mode="lines",
                    name="Portfolio/Watchlist",
                )
            )
            comparison_fig.add_trace(
                go.Scatter(
                    x=compare_df.index,
                    y=compare_df[benchmark_label],
                    mode="lines",
                    name=benchmark_label,
                )
            )
            comparison_fig.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=30, b=10),
                xaxis_title="Date",
                yaxis_title="Index (Start = 100)",
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
            st.caption("Historical comparison only. Not a prediction of future performance.")

    st.subheader("Drawdown history")
    if portfolio_index.empty:
        st.info("Drawdown chart is unavailable because the combined indexed trend is unavailable.")
    else:
        drawdown_series = build_drawdown_series(portfolio_index)
        drawdown_fig = go.Figure()
        drawdown_fig.add_trace(
            go.Scatter(
                x=drawdown_series.index,
                y=drawdown_series.values * 100.0,
                mode="lines",
                name="Drawdown",
            )
        )
        drawdown_fig.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
        )
        st.plotly_chart(drawdown_fig, use_container_width=True)

    st.subheader("Portfolio Scenario Outlook")
    if portfolio_index.empty:
        st.info("Portfolio scenario outlook is unavailable because the combined indexed trend is unavailable.")
    else:
        portfolio_cone_df = build_volatility_cone(portfolio_index, horizon_days=126)
        portfolio_summary_df = build_scenario_summary(portfolio_index)
        if portfolio_cone_df.empty or portfolio_summary_df.empty:
            st.info(
                "Portfolio scenario outlook needs at least 60 daily return observations "
                "from recent price history."
            )
        else:
            annual_volatility = float(portfolio_summary_df["annual_volatility"].iloc[0])
            annual_return = float(portfolio_summary_df["annual_return"].iloc[0])
            scenario_cols = st.columns(3)
            scenario_cols[0].metric("Current index value", f"{float(portfolio_index.iloc[-1]):.2f}")
            scenario_cols[1].metric("Annualized volatility", f"{annual_volatility * 100:.2f}%")
            scenario_cols[2].metric("Risk status", assign_volatility_risk_status(annual_volatility))
            st.caption(f"Annualized return (historical): {annual_return * 100:.2f}%")
            st.plotly_chart(
                _build_scenario_cone_figure(portfolio_cone_df, "Portfolio Volatility Cone"),
                use_container_width=True,
            )
            st.dataframe(
                portfolio_summary_df.rename(
                    columns={
                        "horizon": "Horizon",
                        "best_2sd": "Best (+2 SD)",
                        "normal_upper_1sd": "Normal Upper (+1 SD)",
                        "expected": "Expected / Median",
                        "normal_lower_1sd": "Normal Lower (-1 SD)",
                        "worst_2sd": "Worst (-2 SD)",
                        "annual_return": "Annualized Return",
                        "annual_volatility": "Annualized Volatility",
                    }
                ),
                use_container_width=True,
            )
            for scenario_warning in SCENARIO_WARNINGS:
                st.caption(scenario_warning)
            st.caption("Gemini scenario review: future enhancement")

    st.subheader("Cash deployment planner")
    input_cols = st.columns(2)
    cash_amount = input_cols[0].number_input(
        "Cash amount to review",
        min_value=0.0,
        value=1000.0,
        step=50.0,
    )
    currency_label = input_cols[1].text_input("Currency label", value="SGD").strip().upper() or "SGD"
    deployment_cols = st.columns(2)
    deployment_style = deployment_cols[0].selectbox(
        "Deployment style",
        options=["Conservative", "Balanced", "Opportunistic"],
        index=1,
    )
    include_tactical = deployment_cols[1].toggle("Include tactical bucket candidates", value=False)

    deployment_plan = build_cash_deployment_plan(
        amount=float(cash_amount),
        watchlist=watchlist,
        signals=signals,
        risk=risk,
        deployment_style=deployment_style,
        include_tactical=include_tactical,
    )
    deployment_plan["currency"] = currency_label

    st.write(f"Cash deployment review — {currency_label} {deployment_plan['amount']:,.2f}")
    st.write(f"Style: {deployment_plan['deployment_style']}")
    summary_cols = st.columns(2)
    summary_cols[0].metric(
        "Deploy for review",
        f"{currency_label} {deployment_plan['deploy_now_amount']:,.2f}",
    )
    summary_cols[1].metric(
        "Keep as cash/dry powder",
        f"{currency_label} {deployment_plan['keep_as_cash']:,.2f}",
    )

    allocation_df = pd.DataFrame(deployment_plan["allocation_rows"])
    st.subheader("Allocation review table")
    if allocation_df.empty:
        st.info("No eligible candidates for deployment review.")
    else:
        st.dataframe(
            allocation_df[
                [
                    "ticker",
                    "bucket",
                    "signal",
                    "suggested_amount",
                    "execution_note",
                ]
            ],
            use_container_width=True,
        )

    st.subheader("Tranche review table")
    if allocation_df.empty:
        st.info("No tranche rows to review.")
    else:
        st.dataframe(
            allocation_df[
                [
                    "ticker",
                    "suggested_amount",
                    "tranche_1",
                    "tranche_2",
                    "tranche_3",
                ]
            ],
            use_container_width=True,
        )

    excluded_df = pd.DataFrame(deployment_plan["excluded_rows"])
    st.subheader("Excluded candidates")
    if excluded_df.empty:
        st.write("No exclusions.")
    else:
        st.dataframe(excluded_df, use_container_width=True)

    if deployment_plan["warnings"]:
        st.subheader("Warnings")
        for warning in deployment_plan["warnings"]:
            st.write(f"- {warning}")

    st.info("Manual review required. No trades are executed by this dashboard.")
    st.caption("Gemini allocation review: future enhancement")

# ==========================================================================
# Tab 3 — Watchlist  (investment universe)
# ==========================================================================
with tabs[2]:
    st.header("Watchlist")
    st.caption(
        "The watchlist is the investment universe under monitoring. "
        "It is not the same as your actual portfolio holdings."
    )

    st.subheader("Watchlist table")
    st.dataframe(watchlist, use_container_width=True)

    st.subheader("Signal summary")
    _signal_cols = [
        "ticker",
        "name",
        "type",
        "bucket",
        "latest_price",
        "sma_200",
        "momentum_6m",
        "momentum_3m",
        "signal",
        "signal_reason",
    ]
    if "data_ticker" in signals.columns:
        _signal_cols.insert(1, "data_ticker")
    st.dataframe(
        signals[_signal_cols],
        use_container_width=True,
    )

    st.subheader("Return summary")
    return_summary = build_return_summary(signals, prices)
    display_return_summary = return_summary.rename(
        columns={
            "momentum_3m": "3M momentum",
            "momentum_6m": "6M momentum",
            "return_1m": "1M return",
        }
    )
    st.dataframe(
        display_return_summary[
            [
                "ticker",
                "name",
                "bucket",
                "signal",
                "3M momentum",
                "6M momentum",
                "1M return",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Signal distribution")
    watchlist_signal_distribution = build_signal_distribution(signals)
    st.plotly_chart(
        go.Figure(
            go.Pie(
                labels=watchlist_signal_distribution["signal_group"],
                values=watchlist_signal_distribution["count"],
                hole=0.55,
            )
        ).update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10)),
        use_container_width=True,
    )

# ==========================================================================
# Tab 4 — Ticker Review  (individual asset drill-down)
# ==========================================================================
with tabs[3]:
    st.header("Ticker Review")
    st.caption("Individual asset drill-down — entry guidance and Gemini challenge review.")

    entry_tickers = signals["ticker"].tolist()
    default_entry_index = entry_tickers.index("ES3") if "ES3" in entry_tickers else 0
    selected_entry_ticker = st.selectbox(
        "Select ticker for entry guidance",
        options=entry_tickers,
        index=default_entry_index,
    )
    selected_entry_signal = signals.loc[signals["ticker"] == selected_entry_ticker].iloc[0].to_dict()
    _selected_data_ticker = str(selected_entry_signal.get("data_ticker", "") or "").strip() or None
    _selected_price_history = resolve_price_history(prices, selected_entry_ticker, _selected_data_ticker)
    entry_guidance = calculate_entry_guidance(
        ticker=selected_entry_ticker,
        price_history=_selected_price_history,
        signal_row=selected_entry_signal,
    )

    st.subheader("Ticker profile")
    st.write(f"Selected ticker: **{selected_entry_ticker}**")
    st.write(f"Status: **{entry_guidance['status']}**")
    st.write(f"Signal: **{selected_entry_signal.get('signal', 'N/A')}**")
    if selected_entry_signal.get("signal_reason"):
        st.write(f"Signal reason: {selected_entry_signal['signal_reason']}")
    st.info("Manual review required. No trades are executed by this dashboard.")

    st.subheader("Historical trend")
    ticker_trend_df = build_ticker_trend_dataframe(_selected_price_history)
    if ticker_trend_df.empty:
        st.info("Ticker trend chart is unavailable due to missing or insufficient price history.")
        with st.expander("Show price history diagnostics"):
            st.write(f"Selected ticker: **{selected_entry_ticker}**")
            _resolved_key = _selected_data_ticker or selected_entry_ticker
            st.write(f"Resolved data_ticker: **{_resolved_key}**")
            st.write(f"Available price-history keys: **{len(prices)}**")
            _available_keys = list(prices.keys())[:20]
            st.write(f"Keys (first 20): {_available_keys}")
    else:
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=ticker_trend_df.index,
                y=ticker_trend_df["price"],
                mode="lines",
                name="Price",
            )
        )
        for column, label in [("sma_20", "20D SMA"), ("sma_50", "50D SMA"), ("sma_200", "200D SMA")]:
            trend_fig.add_trace(
                go.Scatter(
                    x=ticker_trend_df.index,
                    y=ticker_trend_df[column],
                    mode="lines",
                    name=label,
                )
            )

        latest_date = ticker_trend_df.index[-1]
        latest_price = float(ticker_trend_df["price"].iloc[-1])
        trend_fig.add_trace(
            go.Scatter(
                x=[latest_date],
                y=[latest_price],
                mode="markers+text",
                text=[f"Latest: {latest_price:.3f}"],
                textposition="top center",
                name="Latest price",
            )
        )
        trend_fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Date",
            yaxis_title="Price",
        )
        st.plotly_chart(trend_fig, use_container_width=True)

    st.subheader("Ticker Scenario Outlook")
    ticker_cone_df = build_volatility_cone(_selected_price_history, horizon_days=126)
    ticker_summary_df = build_scenario_summary(_selected_price_history)
    if ticker_cone_df.empty or ticker_summary_df.empty:
        st.info(
            "Ticker scenario outlook needs at least 60 daily return observations "
            "from recent price history."
        )
    else:
        ticker_annual_return = float(ticker_summary_df["annual_return"].iloc[0])
        ticker_annual_volatility = float(ticker_summary_df["annual_volatility"].iloc[0])
        ticker_scenario_cols = st.columns(4)
        ticker_scenario_cols[0].metric("Current price", format_optional_price(entry_guidance["latest_price"]))
        ticker_scenario_cols[1].metric("Annualized return", f"{ticker_annual_return * 100:.2f}%")
        ticker_scenario_cols[2].metric("Annualized volatility", f"{ticker_annual_volatility * 100:.2f}%")
        ticker_scenario_cols[3].metric(
            "Risk status", assign_volatility_risk_status(ticker_annual_volatility)
        )
        st.plotly_chart(
            _build_scenario_cone_figure(ticker_cone_df, f"{selected_entry_ticker} Volatility Cone"),
            use_container_width=True,
        )
        st.dataframe(
            ticker_summary_df.rename(
                columns={
                    "horizon": "Horizon",
                    "best_2sd": "Best (+2 SD)",
                    "normal_upper_1sd": "Normal Upper (+1 SD)",
                    "expected": "Expected / Median",
                    "normal_lower_1sd": "Normal Lower (-1 SD)",
                    "worst_2sd": "Worst (-2 SD)",
                    "annual_return": "Annualized Return",
                    "annual_volatility": "Annualized Volatility",
                }
            ),
            use_container_width=True,
        )
        for scenario_warning in SCENARIO_WARNINGS:
            st.caption(scenario_warning)
        st.caption("Gemini scenario review: future enhancement")

    st.subheader("Price context")
    price_cols_row1 = st.columns(4)
    price_cols_row1[0].metric("Latest price", format_optional_price(entry_guidance["latest_price"]))
    price_cols_row1[1].metric("20D SMA", format_optional_price(entry_guidance["sma_20"]))
    price_cols_row1[2].metric("50D SMA", format_optional_price(entry_guidance["sma_50"]))
    price_cols_row1[3].metric("200D SMA", format_optional_price(entry_guidance["sma_200"]))
    price_cols_row2 = st.columns(3)
    price_cols_row2[0].metric("ATR / volatility", format_optional_price(entry_guidance["atr_14"]))
    price_cols_row2[1].metric("20D low", format_optional_price(entry_guidance["recent_low_20d"]))
    price_cols_row2[2].metric("20D high", format_optional_price(entry_guidance["recent_high_20d"]))

    st.subheader("Bid zone review")
    st.table(build_bid_zone_table(entry_guidance))
    st.write(f"Preferred order type: **{entry_guidance['preferred_order_type']}**")

    st.subheader("Entry Zone Visual")
    entry_zone_df = build_entry_zone_dataframe(entry_guidance)
    if entry_zone_df.empty:
        st.info("Entry zone visual is unavailable for this ticker.")
    else:
        entry_zone_figure = go.Figure()
        for _, zone_row in entry_zone_df.iterrows():
            entry_zone_figure.add_trace(
                go.Bar(
                    x=[zone_row["display_high"] - zone_row["display_low"]],
                    y=[zone_row["label"]],
                    base=zone_row["display_low"],
                    customdata=[[zone_row["display_high"]]],
                    orientation="h",
                    name=zone_row["label"],
                    hovertemplate="Zone: %{y}<br>Low: %{base:.3f}<br>High: %{customdata[0]:.3f}<extra></extra>",
                    showlegend=False,
                )
            )
        for marker_label, marker_value in [
            ("Latest Price", entry_guidance.get("latest_price")),
            ("20-Day SMA", entry_guidance.get("sma_20")),
            ("50-Day SMA", entry_guidance.get("sma_50")),
            ("200-Day SMA", entry_guidance.get("sma_200")),
        ]:
            if marker_value is not None:
                entry_zone_figure.add_vline(
                    x=float(marker_value),
                    line_dash="dot",
                    annotation_text=marker_label,
                    annotation_position="top",
                )
        entry_zone_figure.update_layout(
            barmode="overlay",
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Price",
            yaxis_title="Entry style",
        )
        st.plotly_chart(entry_zone_figure, use_container_width=True)
        if entry_zone_df["was_normalized"].any():
            st.caption("Bid-zone bounds were normalized for chart display where needed.")

    st.warning(GUARDRAIL_SUMMARY)
    with st.expander("Show detailed guardrails"):
        for warning in entry_guidance["warnings"]:
            st.write(f"- {warning}")

    st.subheader("Manual pre-trade checklist")
    st.write("- Verify live price in Tiger / broker app")
    st.write("- Verify bid/ask spread")
    st.write("- Verify lot size")
    st.write("- Verify position size")
    st.write("- Verify portfolio allocation")
    st.write("- Confirm risk status")
    st.write("- Place order manually only if comfortable")

    st.subheader("Gemini challenge review")
    gemini_prompt = build_gemini_entry_review_prompt(entry_guidance, selected_entry_signal, risk)
    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    st.caption(f"Gemini model: {gemini_model}")
    gemini_review_cache_key = (selected_entry_ticker, gemini_prompt)
    cached_gemini_review = st.session_state.get("gemini_entry_review_cache")
    if cached_gemini_review and cached_gemini_review.get("key") == gemini_review_cache_key:
        gemini_entry_review = cached_gemini_review["value"]
    else:
        gemini_entry_review = review_entry_with_gemini(entry_guidance, selected_entry_signal, risk)
        st.session_state["gemini_entry_review_cache"] = {
            "key": gemini_review_cache_key,
            "value": gemini_entry_review,
        }
    st.info(gemini_entry_review)
    with st.expander("Show Gemini review prompt"):
        st.code(gemini_prompt)

# ==========================================================================
# Tab 5 — AI Review  (AI challenge / review layer)
# ==========================================================================
with tabs[4]:
    st.header("AI Review")
    st.caption(
        "AI signal review layer. "
        "AI explains and challenges — rules decide. User executes manually."
    )

    selected_ai_ticker = st.selectbox("Select ticker for AI review", options=signals["ticker"].tolist())
    selected_ai_signal = signals.loc[signals["ticker"] == selected_ai_ticker].iloc[0].to_dict()
    openai_review = review_with_openai(str(selected_ai_signal))
    gemini_review = review_with_gemini(str(selected_ai_signal))
    consensus = check_consensus(openai_review, gemini_review)
    openai_status = STATUS_NOT_ENABLED if not os.getenv("OPENAI_API_KEY") else STATUS_PLACEHOLDER
    gemini_status = STATUS_NOT_ENABLED if not os.getenv("GEMINI_API_KEY") else STATUS_PLACEHOLDER
    consensus_status = STATUS_INSUFFICIENT_AI if consensus["consensus"] == "insufficient_ai_reviews" else "Pending"
    ai_cols = st.columns(3)
    ai_cols[0].metric("OpenAI review", openai_status)
    ai_cols[1].metric("Gemini review", gemini_status)
    ai_cols[2].metric("Consensus", consensus_status)
    with st.expander("Show raw AI review output"):
        st.json({"openai": openai_review, "gemini": gemini_review, "consensus": consensus})

# ==========================================================================
# Tab 6 — Journal  (manual decision record)
# ==========================================================================
with tabs[5]:
    st.header("Trade Journal")
    st.caption("Manual decision record and execution log. No trades are executed by this dashboard.")

    st.subheader("Manual execution checklist")
    st.write("- Verify rule-based signal reason")
    st.write("- Verify risk status")
    st.write("- Confirm position sizing and exposure")
    st.write("- Execute manually via broker (e.g., Tiger)")

    st.subheader("Decision log placeholder")
    st.info(
        "Future: record ticker, decision, reason, amount, order type, date, and outcome/review notes here."
    )
    st.caption("Trade Journal: future enhancement — manual entry only, no automatic execution.")
