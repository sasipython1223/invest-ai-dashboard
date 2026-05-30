from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.ai_review.consensus_checker import check_consensus
from src.ai_review.gemini_entry_reviewer import build_gemini_entry_review_prompt, review_entry_with_gemini
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
st.warning(GUARDRAIL_SUMMARY)

config = load_config()
watchlist = load_watchlist(config["watchlist_path"])
portfolio = load_portfolio(config["portfolio_path"])
prices = load_prices_for_watchlist(watchlist)
signals = run_signal_engine(watchlist, prices)
risk = evaluate_risk(watchlist, signals)

st.header("1) Decision Center")
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

st.header("2) Watchlist table")
st.dataframe(watchlist, use_container_width=True)

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

st.header("3) Signal summary")
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

st.header("4) Risk alerts")
st.write(f"Risk status: **{risk['status']}**")
if risk.get("reserve_target_weight", 0.0) > 0:
    st.write(f"- Cash reserve (dry powder): {risk['reserve_target_weight']:.2f}% target weight.")
for alert in risk["alerts"]:
    st.write(f"- {alert}")

st.header("5) Portfolio & Cash Deployment Review")
st.subheader("Current portfolio snapshot")
st.dataframe(portfolio, use_container_width=True)

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

st.header("6) AI review placeholder")
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

st.header("7) Entry Guidance")
entry_tickers = signals["ticker"].tolist()
default_entry_index = entry_tickers.index("ES3") if "ES3" in entry_tickers else 0
selected_entry_ticker = st.selectbox(
    "Select ticker for entry guidance",
    options=entry_tickers,
    index=default_entry_index,
)
selected_entry_signal = signals.loc[signals["ticker"] == selected_entry_ticker].iloc[0].to_dict()
entry_guidance = calculate_entry_guidance(
    ticker=selected_entry_ticker,
    price_history=prices.get(selected_entry_ticker, pd.Series(dtype=float)),
    signal_row=selected_entry_signal,
)
st.write(f"Selected ticker: **{selected_entry_ticker}**")
st.write(f"Status: **{entry_guidance['status']}**")
st.write(f"Signal: **{selected_entry_signal.get('signal', 'N/A')}**")
st.info("Manual review required. No trades are executed by this dashboard.")

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
gemini_entry_review = review_entry_with_gemini(entry_guidance, selected_entry_signal, risk)
st.info(gemini_entry_review)
gemini_prompt = build_gemini_entry_review_prompt(entry_guidance, selected_entry_signal, risk)
with st.expander("Show Gemini review prompt"):
    st.code(gemini_prompt)

st.header("8) Trade journal placeholder")
st.write("Manual execution checklist:")
st.write("- Verify rule-based signal reason")
st.write("- Verify risk status")
st.write("- Confirm position sizing and exposure")
st.write("- Execute manually via broker (e.g., Tiger)")
