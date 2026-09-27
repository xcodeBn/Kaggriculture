"""Competing Agent with Town-Conditional Liquidation & Anti-Shock Pricing.

Key strategic insight:
- If an active town shop is consuming a good (drain_rate > 1.0), town consumption
  absorbs supply every 4 turns. Throttling sales allows the market to recover to peak prices.
- If NO town shop consumes the good (drain_rate <= 1.0), excess supply will never
  recover organically; holding inventory risks market front-running and end-of-season decay.
  In this case, execute sales promptly to secure liquidity.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util
_spec = importlib.util.spec_from_file_location("market_model", str(ROOT / "evoexperiments" / "market_model.py"))
market_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(market_model)

calculate_price = market_model.calculate_price
optimal_sale_batch = market_model.optimal_sale_batch
estimate_inventory_from_price = market_model.estimate_inventory_from_price

import submission
from submission import (
    CROP_INFO,
    FINAL_ACTION_STEP,
    SHOP_DEMAND,
    set_policy_config,
)

COMPETING_CONFIG: Dict[str, Any] = {
    "cash_reserve": 483,
    "land_reserve": 846,
    "max_hands": 8,
    "min_cash_for_hand": 0,
    "sell_price_ratio": 0.50,
    "sell_min_price": 1,
    "sell_batch_size": 10,
    "town_demand_weight": 1.0,
    "town_price_weight": 0.3,
    "town_scarcity_weight": 0.097,
    "wheat_ratio": 1.0,
    "carrot_ratio": 1.25,
    "tomato_ratio": 1.0,
    "strawberry_ratio": 1.05,
    "melon_ratio": 0.57,
    "buy_land_enabled": False,
    "hire_enabled": True,
    "max_animals": 8,
    "animal_species": "COW",
}


def get_town_drain_rates(unlocked_shops: list[str]) -> Dict[str, float]:
    """Calculate town consumption rate per day for each product."""
    rates = {item: 1.0 for item in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]}
    for shop in unlocked_shops:
        demands = SHOP_DEMAND.get(shop, {})
        for item, qty in demands.items():
            rates[item] = rates.get(item, 0.0) + (qty * 6.0)
    return rates


class CompetingAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(COMPETING_CONFIG)
        if config:
            self.config.update(config)

    def compute_anti_shock_sales(self, obs: Dict[str, Any], current_step: int) -> List[List[Any]]:
        shed = obs.get("private", {}).get("shed", {})
        prices = obs.get("market", {}).get("prices", {})
        inventories = obs.get("market", {}).get("inventory", {})
        final_turn = current_step >= FINAL_ACTION_STEP
        stagger_phase = current_step >= (FINAL_ACTION_STEP - 48)

        unlocked_shops = obs.get("town", {}).get("unlocked_shops", [])
        drain_rates = get_town_drain_rates(unlocked_shops)

        farms = obs.get("farms", [])
        player = int(obs.get("player", 0))
        farm = farms[player] if player < len(farms) else {}
        animal_count = sum(
            1 for row in farm.get("tiles", []) for tile in row
            if isinstance(tile, dict) and tile.get("animal")
        )
        wheat_reserve = animal_count * 2

        orders = []
        for item, qty in shed.items():
            qty = int(qty)
            if qty <= 0:
                continue

            if item == "WHEAT" and not final_turn:
                qty = max(0, qty - wheat_reserve)
                if qty <= 0:
                    continue

            observed_price = prices.get(item, 0)
            if observed_price <= 0:
                continue

            has_town_drain = drain_rates.get(item, 1.0) > 1.5

            inv = inventories.get(item)
            if inv is None:
                inv = estimate_inventory_from_price(item, observed_price)

            if final_turn:
                batch = qty
            elif has_town_drain:
                # Town is actively draining this item: throttle to let price recover!
                batch = optimal_sale_batch(item, qty, inv, min_marginal_ratio=0.55)
                batch = min(batch, 8 if item in {"MILK", "WOOL", "MELON"} else 12)
            else:
                # No town drain: sell steadily at base batch size to avoid front-running
                batch = min(qty, int(self.config.get("sell_batch_size", 10)))

            if batch > 0:
                orders.append(["SELL", item, batch])

        return orders

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        set_policy_config(self.config)

        base_action = submission.agent(obs)
        farmer_act = base_action.get("farmer", ["PASS"])
        hands_act = base_action.get("hands", [])
        base_market = base_action.get("market", [])

        current_step = int(obs.get("step", 0))
        smart_sales = self.compute_anti_shock_sales(obs, current_step)

        non_sell_orders = [o for o in base_market if isinstance(o, (list, tuple)) and o and o[0] != "SELL"]
        market_orders = (smart_sales + non_sell_orders)[:10]

        return {
            "farmer": farmer_act,
            "hands": hands_act,
            "market": market_orders,
        }


_COMPETING_INSTANCE = CompetingAgent()

def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return _COMPETING_INSTANCE(obs)
