def check_consensus(openai_text: str, gemini_text: str) -> dict[str, str | bool]:
    """Simple placeholder consensus check between reviewer outputs."""
    both_available = "unavailable" not in openai_text.lower() and "unavailable" not in gemini_text.lower()
    return {
        "both_available": both_available,
        "consensus": "pending" if both_available else "insufficient_ai_reviews",
    }
