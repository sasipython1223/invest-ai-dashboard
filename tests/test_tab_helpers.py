from src.ui.tab_helpers import get_tab_labels


def test_get_tab_labels_returns_six_tabs():
    labels = get_tab_labels()
    assert len(labels) == 6


def test_get_tab_labels_has_expected_names():
    labels = get_tab_labels()
    assert labels == [
        "Overview",
        "Portfolio",
        "Watchlist",
        "Ticker Review",
        "AI Review",
        "Journal",
    ]


def test_get_tab_labels_returns_list_of_strings():
    labels = get_tab_labels()
    assert isinstance(labels, list)
    assert all(isinstance(label, str) for label in labels)
