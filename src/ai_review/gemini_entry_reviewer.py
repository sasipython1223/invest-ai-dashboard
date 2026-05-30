from __future__ import annotations

import os

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None

NOT_ENABLED_MESSAGE = "Gemini entry review is not enabled. Set GEMINI_API_KEY to activate independent review."
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def build_gemini_entry_review_prompt(
    entry_guidance: dict,
    signal_row: dict,
    risk: dict,
) -> str:
    return "\n".join(
        [
            "Act as an independent investment risk reviewer.",
            "You are reviewing deterministic ETF entry-guidance output.",
            "Use only the supplied data. Do not invent prices or market data.",
            "Do not provide personal financial advice.",
            "Do not generate raw buy/sell signals.",
            "Do not override deterministic calculations.",
            "Do not place or execute trades, and do not suggest broker API or Tiger API automation.",
            "",
            "Review data:",
            f"entry_guidance={entry_guidance}",
            f"signal_row={signal_row}",
            f"risk={risk}",
            "",
            "Output required:",
            "1. Do you agree with the entry guidance?",
            "2. What risks are missing?",
            "3. Is the proposed bid zone too aggressive?",
            "4. What should the user verify in the broker app before placing a manual limit order?",
            "5. Final status: Accept / Accept with caution / Defer / Reject",
        ]
    )


def review_entry_with_gemini(
    entry_guidance: dict,
    signal_row: dict,
    risk: dict,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return NOT_ENABLED_MESSAGE

    if genai is None:
        return "Gemini entry review unavailable: google-genai package is not installed."

    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    prompt = build_gemini_entry_review_prompt(entry_guidance, signal_row, risk)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
    except Exception as exc:  # pragma: no cover
        return (
            "Gemini entry review failed: "
            f"{exc.__class__.__name__} from Gemini API. "
            "Check GEMINI_MODEL, API key/project access, and quota. "
            "No API key was printed."
        )

    if getattr(response, "text", None):
        return response.text
    return "Gemini entry review failed: empty response."
