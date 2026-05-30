from src.ai_review.gemini_entry_reviewer import (
    build_gemini_entry_review_prompt,
    review_entry_with_gemini,
)


def test_gemini_entry_prompt_uses_supplied_data_only():
    entry_guidance = {"ticker": "ES3", "latest_price": 5.095, "normal_bid_zone": "5.020 to 5.060"}
    signal_row = {"signal": "Hold / Buy Candidate", "signal_reason": "Price above 200D SMA"}
    risk = {"status": "OK", "alerts": ["No rule-based risk alerts."]}

    prompt = build_gemini_entry_review_prompt(entry_guidance, signal_row, risk)

    assert "entry_guidance={'ticker': 'ES3', 'latest_price': 5.095, 'normal_bid_zone': '5.020 to 5.060'}" in prompt
    assert "signal_row={'signal': 'Hold / Buy Candidate', 'signal_reason': 'Price above 200D SMA'}" in prompt
    assert "risk={'status': 'OK', 'alerts': ['No rule-based risk alerts.']}" in prompt


def test_gemini_prompt_contains_guardrails_against_invention_and_execution():
    prompt = build_gemini_entry_review_prompt({"ticker": "ES3"}, {"signal": "Watch"}, {"status": "REVIEW"})

    assert "Use only the supplied data. Do not invent prices or market data." in prompt
    assert "Do not place or execute trades" in prompt
    assert "Do not override deterministic calculations." in prompt
    assert "Final status: Accept / Accept with caution / Defer / Reject" in prompt


def test_gemini_entry_review_unavailable_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    response = review_entry_with_gemini({"ticker": "ES3"}, {"signal": "Watch"}, {"status": "REVIEW"})

    assert response == "Gemini entry review unavailable: set GEMINI_API_KEY to enable."
