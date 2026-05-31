def has_price_data(price_map: object) -> bool:
    """Return True if *price_map* is a non-empty dict containing at least one
    non-empty price series.

    ``prices`` returned by :func:`load_prices_for_watchlist` is a
    ``dict[str, pd.DataFrame]``, so ``.empty`` on the dict itself raises
    ``AttributeError``.  This helper handles that safely and also tolerates
    other iterable value types.
    """
    if not isinstance(price_map, dict):
        return False
    for value in price_map.values():
        if hasattr(value, "empty") and not value.empty:
            return True
        try:
            if len(value) > 0:
                return True
        except TypeError:
            continue
    return False
