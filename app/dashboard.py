from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.ai_review.consensus_checker import check_consensus
from src.ai_review.gemini_reviewer import review_with_gemini
from src.ai_review.openai_reviewer import review_with_openai
from src.data.portfolio_loader import load_portfolio
from src.data.price_loader import load_prices_for_watchlist
from src.dashboard_summary import MANUAL_REVIEW_NOTE, build_action_items, count_signals
from src.data.watchlist_loader import load_watchlist
from src.risk.risk_engine import evaluate_risk
from src.signals.signal_engine import run_signal_engine
from src.utils.config import load_config


st.set_page_config(page_title="invest-ai-dashboard", layout="wide")
st.title("invest-ai-dashboard")
st.caption("Rules decide. AI explains. User executes manually.")

st.warning(
    "Educational prototype only. Not financial advice. "
    "No auto-trading. All trades must be manually reviewed and executed by the user."
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
metric_columns[4].metric("Cash reserve target weight", f"{risk['reserve_target_weight']:.2f}%")

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
selected_ticker = st.selectbox("Select ticker for AI review", options=signals["ticker"].tolist())
selected_signal = signals.loc[signals["ticker"] == selected_ticker].iloc[0].to_dict()
openai_review = review_with_openai(str(selected_signal))
gemini_review = review_with_gemini(str(selected_signal))
consensus = check_consensus(openai_review, gemini_review)
st.write({"openai": openai_review, "gemini": gemini_review, "consensus": consensus})

st.header("7) Trade journal placeholder")
st.write("Manual execution checklist:")
st.write("- Verify rule-based signal reason")
st.write("- Verify risk status")
st.write("- Confirm position sizing and exposure")
st.write("- Execute manually via broker (e.g., Tiger)")
