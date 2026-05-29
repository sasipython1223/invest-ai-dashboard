SYSTEM_PROMPT = (
    "You are an investment review assistant. Explain and challenge existing "
    "rule-based signals. Never generate standalone buy/sell calls."
)


def build_review_prompt(signal_row: dict) -> str:
    return (
        "Review this rule-based signal and provide risks, assumptions, and "
        f"questions for manual review: {signal_row}"
    )
