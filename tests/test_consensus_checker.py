from src.ai_review.consensus_checker import check_consensus


def test_consensus_checker_insufficient_when_placeholder_unavailable():
    result = check_consensus(
        "OpenAI reviewer unavailable: set OPENAI_API_KEY to enable.",
        "Gemini reviewer unavailable: set GEMINI_API_KEY to enable.",
    )

    assert result["both_available"] is False
    assert result["consensus"] == "insufficient_ai_reviews"


def test_consensus_checker_pending_when_both_available():
    result = check_consensus("review text A", "review text B")

    assert result["both_available"] is True
    assert result["consensus"] == "pending"
