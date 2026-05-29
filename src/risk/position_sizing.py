def suggest_position_size(
    portfolio_value: float,
    target_weight: float,
    max_single_position_weight: float = 20.0,
) -> float:
    capped_weight = min(float(target_weight), float(max_single_position_weight))
    return portfolio_value * capped_weight / 100.0
