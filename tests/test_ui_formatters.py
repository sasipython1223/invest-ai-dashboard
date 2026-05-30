from src.ui.formatters import build_bid_zone_table, format_optional_price


def test_format_optional_price_handles_none():
    assert format_optional_price(None) == "N/A"


def test_build_bid_zone_table_has_expected_rows():
    table = build_bid_zone_table(
        {
            "aggressive_bid_zone": "5.081 to 5.095",
            "normal_bid_zone": "5.068 to 5.071",
            "conservative_bid_zone": "5.009 to 5.009",
        }
    )

    assert table["Entry style"].tolist() == ["Aggressive", "Normal", "Conservative"]
    assert table["Bid zone"].tolist() == ["5.081 to 5.095", "5.068 to 5.071", "5.009 to 5.009"]
