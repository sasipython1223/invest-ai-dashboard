import os

from src.ai_review.prompts import SYSTEM_PROMPT


def review_with_openai(prompt: str) -> str:
    """Placeholder for future OpenAI integration."""
    if not os.getenv("OPENAI_API_KEY"):
        return "OpenAI reviewer unavailable: set OPENAI_API_KEY to enable."
    return "OpenAI reviewer placeholder response."
