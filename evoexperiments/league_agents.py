"""League Archetypes for Co-Evolutionary Multi-Agent Training.

Defines two distinct competing playstyles built on top of EvolvedAgent:
1. SwarmPlanter: Aggressive multi-quadrant land expansion, 10 hands, large strawberry/melon crop coverage, backed by 12 animals.
2. DairyBaron: Central livestock corridor dominance (up to 14 cows/sheep), high-margin milk & wool focus, supplemental feed trading.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import submission
from submission import (
    CROP_INFO, distance, next_step_toward, count_empty_tiles, unlocked_empty_tiles,
    find_weeds, count_crop, animal_goals, ANIMAL_INFO, sell_orders, animal_hand_actions,
    find_harvest_target, find_urgent_water_target, final_turn_liquidation,
    FINAL_ACTION_STEP, farmer_basic_action, next_land_cost, plant_tiles,
    EvolvedAgent, KaggricultureAgent
)

SWARM_PLANTER_BASE: Dict[str, Any] = {
    "name": "SwarmPlanter",
    "cash_reserve": 180,
    "land_reserve": 1300,
    "buy_land_cash_multiplier": 1.15,
    "buy_land_enabled": True,
    "expand_day_first": 10,
    "expand_day_second": 13,
    "max_hands": 10,
    "min_cash_for_hand": 50,
    "max_animals": 12,
    "cow_goal": 6,
    "sheep_goal": 6,
    "strawberry_ratio": 2.2,
    "melon_ratio": 1.4,
    "carrot_ratio": 0.8,
    "wheat_ratio": 0.5,
    "sell_price_ratio": 0.25,
    "sell_batch_size": 10,
}

DAIRY_BARON_BASE: Dict[str, Any] = {
    "name": "DairyBaron",
    "cash_reserve": 250,
    "land_reserve": 2000,
    "buy_land_cash_multiplier": 1.20,
    "buy_land_enabled": True,
    "expand_day_first": 12,
    "expand_day_second": 16,
    "max_hands": 9,
    "min_cash_for_hand": 60,
    "max_animals": 14,
    "cow_goal": 8,
    "sheep_goal": 6,
    "strawberry_ratio": 1.0,
    "melon_ratio": 0.8,
    "carrot_ratio": 0.6,
    "wheat_ratio": 1.0,
    "sell_price_ratio": 0.30,
    "sell_batch_size": 10,
}


class LeagueBaseAgent(EvolvedAgent):
    """Base class providing shared multi-quadrant and parallel execution capabilities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.sales: Dict[str, int] = {}
        self.land_events: List[Tuple[int, int, float, int]] = []

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        action = super().__call__(obs)

        # Record sales for tournament analysis
        for m in action.get("market", []):
            if m and m[0] == "SELL":
                self.sales[m[1]] = self.sales.get(m[1], 0) + m[2]

        return action


class SwarmPlanterAgent(LeagueBaseAgent):
    """Archetype A: Multi-quadrant Swarm Planter with 12-animal baseline and 10-hand field army."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(SWARM_PLANTER_BASE)
        if config:
            cfg.update(config)
        super().__init__(cfg)


class DairyBaronAgent(LeagueBaseAgent):
    """Archetype B: Central Livestock Corridor Baron with 14 cows/sheep and market protection."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(DAIRY_BARON_BASE)
        if config:
            cfg.update(config)
        super().__init__(cfg)

