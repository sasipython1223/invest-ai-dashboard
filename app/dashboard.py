from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.ai_review.consensus_checker import check_consensus
from src.ai_review.gemini_entry_reviewer import build_gemini_entry_review_prompt
from src.ai_review.gemini_reviewer import review_with_gemini
from src.ai_review.openai_reviewer import review_with_openai
from src.data.portfolio_loader import load_portfolio
from src.data.price_loader import load_prices_for_watchlist
from src.dashboard_summary import MANUAL_REVIEW_NOTE, build_action_items, count_signals
from src.data.watchlist_loader import load_watchlist
from src.entries.entry_guidance import calculate_entry_guidance
from src.risk.risk_engine import evaluate_risk
from src.signals.signal_engine import run_signal_engine
from src.ui.formatters import build_bid_zone_table, format_optional_price
from src.utils.config import load_config


st.set_page_config(page_title="invest-ai-dashboard", layout="wide")
st.title("invest-ai-dashboard")
st.caption("Rules decide. AI explains. User executes manually.")

st.warning(
    "Educational prototype only. Not financial advice. "
    "No auto-trading or broker/Tiger API execution. "
    "All trades must be manually reviewed and executed by the user."
)

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

st.info(MANUAL_REVIEW_NOTE)
st.subheader("Today's Action List")
for action_item in build_action_items(signals, risk):
    st.write(action_item)

st.header("2) Watchlist table")
st.dataframe(watchlist, use_container_width=True)

st.header("3) Signal summary")
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

st.header("5) Portfolio placeholder")
st.dataframe(portfolio, use_container_width=True)

st.header("6) AI review placeholder")
selected_ai_ticker = st.selectbox("Select ticker for AI review", options=signals["ticker"].tolist())
selected_ai_signal = signals.loc[signals["ticker"] == selected_ai_ticker].iloc[0].to_dict()
openai_review = review_with_openai(str(selected_ai_signal))
gemini_review = review_with_gemini(str(selected_ai_signal))
consensus = check_consensus(openai_review, gemini_review)
openai_status = "Not enabled" if "unavailable" in openai_review.lower() else "Placeholder"
gemini_status = "Not enabled" if "unavailable" in gemini_review.lower() else "Placeholder"
consensus_status = "Insufficient AI reviews" if consensus["consensus"] == "insufficient_ai_reviews" else "Pending"
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

st.warning(
    "Educational prototype only. Not financial advice. No auto-trading, broker API, or Tiger API execution."
)
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
if not os.getenv("GEMINI_API_KEY"):
    st.info("Gemini entry review is not enabled. Set GEMINI_API_KEY to activate independent review.")
else:
    st.info("Gemini entry review placeholder active. Real API integration pending.")
with st.expander("Show Gemini review prompt"):
    st.code(gemini_prompt)

st.header("8) Trade journal placeholder")
st.write("Manual execution checklist:")
st.write("- Verify rule-based signal reason")
st.write("- Verify risk status")
st.write("- Confirm position sizing and exposure")
st.write("- Execute manually via broker (e.g., Tiger)")
