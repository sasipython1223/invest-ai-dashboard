from __future__ import annotations

import os


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
    prompt = build_gemini_entry_review_prompt(entry_guidance, signal_row, risk)
    if not os.getenv("GEMINI_API_KEY"):
        return "Gemini entry review unavailable: set GEMINI_API_KEY to enable."
    return f"Gemini entry review placeholder response for prompt: {prompt}"
