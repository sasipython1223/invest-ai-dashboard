from types import SimpleNamespace

import src.ai_review.gemini_entry_reviewer as gemini_entry_reviewer
from src.ai_review.gemini_entry_reviewer import build_gemini_entry_review_prompt, review_entry_with_gemini


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

    assert response == "Gemini entry review is not enabled. Set GEMINI_API_KEY to activate independent review."


def test_gemini_entry_review_calls_client_with_configured_model(monkeypatch):
    captured = {}

    class FakeModels:
        def generate_content(self, *, model, contents):
            captured["model"] = model
            captured["contents"] = contents
            return SimpleNamespace(text="Accept with caution")

    class FakeClient:
        def __init__(self, *, api_key):
            captured["api_key"] = api_key
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-model")
    monkeypatch.setattr(gemini_entry_reviewer, "genai", SimpleNamespace(Client=FakeClient))

    response = review_entry_with_gemini({"ticker": "ES3"}, {"signal": "Watch"}, {"status": "REVIEW"})

    assert response == "Accept with caution"
    assert captured["api_key"] == "test-api-key"
    assert captured["model"] == "gemini-custom-model"
    assert "entry_guidance={'ticker': 'ES3'}" in captured["contents"]


def test_gemini_entry_review_failure_returns_safe_error(monkeypatch):
    class FailingModels:
        def generate_content(self, *, model, contents):
            raise RuntimeError("request failed: test-api-key")

    class FailingClient:
        def __init__(self, *, api_key):
            self.models = FailingModels()

    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(gemini_entry_reviewer, "genai", SimpleNamespace(Client=FailingClient))

    response = review_entry_with_gemini({"ticker": "ES3"}, {"signal": "Watch"}, {"status": "REVIEW"})

    assert response == "Gemini entry review failed (RuntimeError). Please try again."
    assert "test-api-key" not in response
