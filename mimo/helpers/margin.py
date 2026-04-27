def required_margin(
    *,
    price: float,
    lots: float,
    contract_size: float,
    leverage: float,
    margin_rate: float | None = None
) -> float:

    notional = price * contract_size * lots
    if margin_rate is not None:
        return notional * margin_rate

    leverage = max(1.0, leverage)
    return notional / leverage