import ast
import inspect

import pandas as pd

from src.ui.decision_center import (
    build_gauge_explanation,
    calculate_cash_reserve_scenario,
    get_current_reserve_target,
    group_tickers_by_signal,
)


def test_group_tickers_by_signal_groups_expected_tickers():
    signals = pd.DataFrame(
        [
            {"ticker": "VWRA", "signal": "Hold / Buy Candidate"},
            {"ticker": "CSPX", "signal": "Hold / Buy Candidate"},
            {"ticker": "DBS", "signal": "Watch"},
            {"ticker": "MSFT", "signal": "Reduce / Avoid"},
            {"ticker": "CASH", "signal": "Cash Reserve / No Signal"},
        ]
    )

    grouped = group_tickers_by_signal(signals)

    assert grouped["hold_buy"] == ["VWRA", "CSPX"]
    assert grouped["watch"] == ["DBS"]
    assert grouped["reduce_avoid"] == ["MSFT"]
    assert grouped["cash"] == ["CASH"]


def test_watchlist_health_explanation_includes_hold_buy_and_reduce_avoid_tickers():
    signals = pd.DataFrame(
        [
            {"ticker": "VWRA", "signal": "Hold / Buy Candidate"},
            {"ticker": "CSPX", "signal": "Hold / Buy Candidate"},
            {"ticker": "MSFT", "signal": "Reduce / Avoid"},
            {"ticker": "CASH", "signal": "Cash Reserve / No Signal"},
        ]
    )

    explanation = build_gauge_explanation(
        signals=signals,
        health_score=88,
        urgency_score=40,
        reserve_target_weight=12.0,
    )

    assert "88 = " in explanation["health_explanation"]
    assert explanation["grouped_tickers"]["hold_buy"] == ["VWRA", "CSPX"]
    assert explanation["grouped_tickers"]["reduce_avoid"] == ["MSFT"]


def test_action_urgency_explanation_includes_reduce_avoid_drivers():
    signals = pd.DataFrame(
        [
            {"ticker": "VWRA", "signal": "Hold / Buy Candidate"},
            {"ticker": "MSFT", "signal": "Reduce / Avoid"},
        ]
    )

    explanation = build_gauge_explanation(
        signals=signals,
        health_score=50,
        urgency_score=40,
        reserve_target_weight=12.0,
    )

    assert "MSFT" in explanation["urgency_explanation"]
    assert explanation["urgency_drivers"][0] == "Reduce / Avoid: MSFT"
    assert explanation["urgency_drivers"][1] == "Cash reserve target: 12.00%"
    assert explanation["urgency_drivers"][2] == "No automatic execution"


def test_cash_reserve_scenario_compares_current_and_test_targets():
    scenario = calculate_cash_reserve_scenario(
        review_amount=1000.0,
        current_reserve_pct=12.0,
        test_reserve_pct=20.0,
    ).set_index("scenario")

    assert scenario.loc["Current target", "keep_as_cash"] == 120.0
    assert scenario.loc["Current target", "deploy_for_review"] == 880.0
    assert scenario.loc["Test target", "keep_as_cash"] == 200.0
    assert scenario.loc["Test target", "deploy_for_review"] == 800.0


def test_get_current_reserve_target_does_not_mutate_watchlist_dataframe():
    watchlist = pd.DataFrame(
        [
            {"ticker": "VWRA", "target_weight": 40.0},
            {"ticker": "CASH", "target_weight": 12.0},
        ]
    )
    original = watchlist.copy(deep=True)

    current_target = get_current_reserve_target(watchlist)
    _ = calculate_cash_reserve_scenario(
        review_amount=1000.0,
        current_reserve_pct=current_target,
        test_reserve_pct=20.0,
    )

    pd.testing.assert_frame_equal(watchlist, original)


def test_decision_center_module_does_not_import_broker_or_trading_modules():
    import src.ui.decision_center as decision_center

    module_ast = ast.parse(inspect.getsource(decision_center))

    imported_modules: set[str] = set()
    called_names: set[str] = set()
    for node in ast.walk(module_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.lower())
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.lower())
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id.lower())
            if isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr.lower())

    assert all("tiger" not in module_name for module_name in imported_modules)
    assert all("broker" not in module_name for module_name in imported_modules)
    assert "execute_trade" not in called_names
    assert "place_order" not in called_names
