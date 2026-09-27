"""Closed-form market price curve and marginal revenue solver for Kaggriculture.

Implements the official price curve and per-resource dynamics as specified
in README.md. Used to compute price impact and optimal liquidation quantities.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

# Official market parameters from game specification
MARKET_PARAMS: Dict[str, Dict[str, Any]] = {
    "WHEAT": {
        "base": 25, "I0": 10000, "T": 400,
        "below_func": "sqrt", "below_target": 0.80,
        "above_func": "log", "above_target": 0.20
    },
    "CARROT": {
        "base": 35, "I0": 10000, "T": 450,
        "below_func": "hinge", "below_target": 1.00,
        "above_func": "sqrt", "above_target": 0.70
    },
    "TOMATO": {
        "base": 60, "I0": 10000, "T": 200,
        "below_func": "hinge", "below_target": 0.40,
        "above_func": "sqrt", "above_target": 0.60
    },
    "STRAWBERRY": {
        "base": 120, "I0": 10000, "T": 100,
        "below_func": "sqrt", "below_target": 0.70,
        "above_func": "linear", "above_target": 1.60
    },
    "MELON": {
        "base": 250, "I0": 10000, "T": 300,
        "below_func": "log", "below_target": 0.20,
        "above_func": "sq", "above_target": 3.60
    },
    "EGG": {
        "base": 50, "I0": 10000, "T": 332,
        "below_func": "hinge", "below_target": 0.40,
        "above_func": "log", "above_target": 0.20
    },
    "MILK": {
        "base": 160, "I0": 10000, "T": 122,
        "below_func": "sqrt", "below_target": 0.60,
        "above_func": "linear", "above_target": 1.60
    },
    "WOOL": {
        "base": 200, "I0": 10000, "T": 105,
        "below_func": "log", "below_target": 0.20,
        "above_func": "sq", "above_target": 3.20
    },
    "FERTILIZER": {
        "base": 100, "I0": 10000, "T": 200,
        "below_func": "linear", "below_target": 0.40,
        "above_func": "linear", "above_target": 0.40
    },
}


def _eval_func(func_name: str, x: float, T: float) -> float:
    """Evaluate shape function f(x)."""
    if x <= 0:
        return 0.0
    if func_name == "linear":
        return float(x)
    elif func_name == "sq":
        return float(x * x)
    elif func_name == "sqrt":
        return math.sqrt(x)
    elif func_name == "log":
        return math.log(1.0 + x)
    elif func_name == "log10":
        return math.log10(1.0 + x)
    elif func_name == "hinge":
        u = x / T
        return u + 8.0 * (max(0.0, u - 1.0) ** 2)
    raise ValueError(f"Unknown shape function: {func_name}")


def calculate_price(item: str, inv: float) -> int:
    """Calculate the exact unfloored/floored rounded price at market inventory inv."""
    params = MARKET_PARAMS.get(item)
    if not params:
        return 1
    base = params["base"]
    I0 = params["I0"]
    T = params["T"]
    diff = inv - I0

    if abs(diff) < 1e-9:
        return int(base)

    if diff < 0:
        # Scarcity side: inv < I0 -> price up
        sign = 1.0
        func_name = params["below_func"]
        target = params["below_target"]
        x = -diff
    else:
        # Glut side: inv > I0 -> price down
        sign = -1.0
        func_name = params["above_func"]
        target = params["above_target"]
        x = diff

    f_T = _eval_func(func_name, T, T)
    amp = target * base / f_T if f_T > 0 else 0.0
    raw_price = base + sign * amp * _eval_func(func_name, x, T)
    
    # Floored at $1, rounded to nearest dollar
    return max(1, round(raw_price))


def estimate_inventory_from_price(item: str, observed_price: int, hint_above: bool = True) -> int:
    """Invert the price function to infer current market inventory."""
    params = MARKET_PARAMS.get(item)
    if not params:
        return 10000
    I0 = params["I0"]
    base = params["base"]
    T = params["T"]

    if observed_price == base:
        return I0

    if observed_price > base or (not hint_above and observed_price >= base):
        # Scarcity side
        func_name = params["below_func"]
        target = params["below_target"]
        f_T = _eval_func(func_name, T, T)
        amp = target * base / f_T
        needed_fx = (observed_price - base) / amp
        # Binary search for x
        low, high = 0.0, float(I0)
        for _ in range(30):
            mid = (low + high) / 2.0
            if _eval_func(func_name, mid, T) < needed_fx:
                low = mid
            else:
                high = mid
        return max(0, int(round(I0 - low)))
    else:
        # Glut side
        func_name = params["above_func"]
        target = params["above_target"]
        f_T = _eval_func(func_name, T, T)
        amp = target * base / f_T
        needed_fx = (base - observed_price) / amp
        low, high = 0.0, 50000.0
        for _ in range(35):
            mid = (low + high) / 2.0
            if _eval_func(func_name, mid, T) < needed_fx:
                low = mid
            else:
                high = mid
        return int(round(I0 + low))


def compute_expected_proceeds(item: str, start_inv: int, quantity: int) -> Tuple[int, int]:
    """Compute total expected revenue and final price if selling quantity units sequentially."""
    total_revenue = 0
    cur_inv = start_inv
    for _ in range(quantity):
        p = calculate_price(item, cur_inv)
        total_revenue += p
        # When price reaches $1 floor, units are bought but not added to inventory
        if p > 1:
            cur_inv += 1
    final_price = calculate_price(item, cur_inv)
    return total_revenue, final_price


def optimal_sale_batch(
    item: str,
    shed_qty: int,
    start_inv: int,
    min_marginal_ratio: float = 0.65,
    final_turn: bool = False
) -> int:
    """Calculate the optimal quantity to sell this turn.

    Balances revenue capture against catastrophic price shock (driving price to $1).
    """
    if shed_qty <= 0:
        return 0
    if final_turn:
        # On the final turn, any unsold goods expire worthlessly. Sell all.
        return shed_qty

    params = MARKET_PARAMS.get(item)
    if not params:
        return min(shed_qty, 10)

    base = params["base"]
    cutoff_price = max(2, int(round(base * min_marginal_ratio)))

    qty = 0
    cur_inv = start_inv
    for _ in range(shed_qty):
        p = calculate_price(item, cur_inv)
        if p < cutoff_price:
            break
        qty += 1
        if p > 1:
            cur_inv += 1

    return max(0, qty)
