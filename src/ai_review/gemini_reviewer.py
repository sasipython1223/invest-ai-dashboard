import os


def review_with_gemini(prompt: str) -> str:
    """Placeholder for future Gemini integration."""
    if not os.getenv("GEMINI_API_KEY"):
        return "Gemini reviewer unavailable: set GEMINI_API_KEY to enable."
    return f"Gemini reviewer placeholder response for prompt: {prompt}"
