"""UI helper utilities for the tab-based dashboard layout."""


def get_tab_labels() -> list[str]:
    """Return the ordered list of dashboard tab labels."""
    return [
        "Overview",
        "Portfolio",
        "Watchlist",
        "Ticker Review",
        "AI Review",
        "Journal",
    ]
