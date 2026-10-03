"""Double Agent Policy Engine for Kaggriculture.

Autonomous, market-adaptive strategic engine designed for
Double-Agent Self-Play and Co-Evolution.
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import submission
from submission import (
    ANIMAL_INFO,
    CROP_INFO,
    FINAL_ACTION_STEP,
    LAND_COSTS,
    MARKET_PARAMS,
    SHOP_DEMAND,
    calculate_price,
    count_crop,
    count_empty_tiles,
    distance,
    estimate_inventory_from_price,
    final_turn_liquidation,
    find_harvest_target,
    find_urgent_water_target,
    find_weeds,
    next_land_cost,
    next_step_toward,
    optimal_sale_batch,
    plant_tiles,
    rank_sell_orders,
    unlocked_empty_tiles,
)

DEFAULT_DOUBLE_AGENT_CONFIG: Dict[str, Any] = {
    # Livestock Goals & Allocation
    "base_sheep_goal": 8,
    "base_cow_goal": 6,
    "day0_hires": 5,
    "day0_animal": "SHEEP",  # or COW
    # Workforce Schedule
    "max_hands_p1": 5,   # Days 0-3
    "max_hands_p2": 8,   # Days 4-8
    "max_hands_p3": 10,  # Days 9-22
    "max_hands_p4": 7,   # Days 23-28
    "min_cash_for_hand": 30,
    # Land Expansion
    "expand_day_first": 7,
    "expand_day_second": 13,
    "land_reserve": 600,
    "buy_land_cash_multiplier": 1.10,
    "buy_land_enabled": True,
    # Crops
    "max_wheat_tiles": 2,
    "melon_weight": 2.2,
    "strawberry_weight": 2.2,
    "carrot_weight": 0.8,
    # Anti-Shock Selling
    "sell_price_ratio": 0.50,
    "sell_batch_size": 8,
    "drop_threshold": 2,
}


def get_town_drain(unlocked_shops: List[str]) -> Dict[str, float]:
    """Calculate town consumption rate per day for each product."""
    rates = {item: 1.0 for item in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]}
    for shop in unlocked_shops:
        demands = SHOP_DEMAND.get(shop, {})
        for item, qty in demands.items():
            rates[item] = rates.get(item, 0.0) + (qty * 6.0)
    return rates


def get_unlocked_central_tiles(farm: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return empty unlocked tiles ordered by proximity to board center (4.5, 4.5)."""
    empty = unlocked_empty_tiles(farm)
    empty.sort(key=lambda p: (abs(p[0] - 4.5) + abs(p[1] - 4.5), p[0] + p[1]))
    return empty


class DoubleAgent:
    """Strategic Double-Agent Policy Engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(DEFAULT_DOUBLE_AGENT_CONFIG)
        if config:
            self.config.update(config)

    def get_max_hands(self, day: int) -> int:
        if day < 4:
            return int(self.config.get("max_hands_p1", 5))
        elif day < 9:
            return int(self.config.get("max_hands_p2", 8))
        elif day < 23:
            return int(self.config.get("max_hands_p3", 10))
        elif day < 28:
            return int(self.config.get("max_hands_p4", 7))
        else:
            return 5

    def get_dynamic_animal_goals(self, obs: Dict[str, Any], farm: Dict[str, Any]) -> Tuple[int, int]:
        """Dynamically modulate Sheep vs Cow targets based on town shop drain and prices."""
        unlocked_shops = obs.get("town", {}).get("unlocked_shops", [])
        drain = get_town_drain(unlocked_shops)
        prices = obs.get("market", {}).get("prices", {})
        wool_p = prices.get("WOOL", 200)
        milk_p = prices.get("MILK", 160)

        cows_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "COW")
        sheep_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "SHEEP")

        base_sheep = int(self.config.get("base_sheep_goal", 8))
        base_cow = int(self.config.get("base_cow_goal", 6))

        # Has active wool drain
        has_yarn = drain.get("WOOL", 1.0) > 1.5
        has_dairy = drain.get("MILK", 1.0) > 1.5

        if has_yarn and not has_dairy:
            # Heavy sheep focus
            target_sheep = max(sheep_now, 13)
            target_cow = max(cows_now, 1)
        elif has_dairy and not has_yarn:
            # Heavy cow focus
            target_sheep = max(sheep_now, 2 if wool_p >= 120 else 0)
            target_cow = max(cows_now, 12)
        elif has_yarn and has_dairy:
            # Dual livestock empire
            target_sheep = max(sheep_now, 8)
            target_cow = max(cows_now, 6)
        else:
            # Neither shop yet: balanced start
            target_sheep = max(sheep_now, base_sheep if wool_p >= 100 else 4)
            target_cow = max(cows_now, base_cow if milk_p >= 90 else 4)

        return target_sheep, target_cow

    def choose_strategic_crop(self, obs: Dict[str, Any], farm: Dict[str, Any], day: int) -> Optional[str]:
        if day >= 28:
            return None
        if day >= 26:
            return "WHEAT"
        if day >= 24:
            return "CARROT"

        money = farm.get("money", 0)
        current_wheat = count_crop(farm, "WHEAT")
        max_wheat = int(self.config.get("max_wheat_tiles", 2))

        # Only plant wheat if we are very broke (< $400) or under feed buffer
        if money < 400 and current_wheat < max_wheat:
            return "WHEAT"

        prices = obs.get("market", {}).get("prices", {})
        melon_p = prices.get("MELON", 250)
        straw_p = prices.get("STRAWBERRY", 120)
        carrot_p = prices.get("CARROT", 35)

        m_wt = float(self.config.get("melon_weight", 2.2))
        s_wt = float(self.config.get("strawberry_weight", 2.2))
        c_wt = float(self.config.get("carrot_weight", 0.8))

        melon_score = (melon_p / 250.0) * m_wt
        straw_score = (straw_p / 120.0) * s_wt
        carrot_score = (carrot_p / 35.0) * c_wt

        melon_cnt = count_crop(farm, "MELON")
        straw_cnt = count_crop(farm, "STRAWBERRY")

        if day <= 16:
            if melon_score >= straw_score and melon_cnt < 16 and melon_p >= 130:
                return "MELON"
            if straw_p >= 50 and straw_cnt < 30:
                return "STRAWBERRY"
            if melon_cnt < 12 and melon_p >= 110:
                return "MELON"
            return "CARROT"
        elif day <= 22:
            if straw_p >= 55 and straw_cnt < 28:
                return "STRAWBERRY"
            return "CARROT"
        else:
            return "CARROT"

    def plan_market_orders(self, obs: Dict[str, Any], farm: Dict[str, Any]) -> List[List[Any]]:
        market_orders: List[List[Any]] = []
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        money = int(farm.get("money", 0))
        private = obs.get("private", {})
        shed = private.get("shed", {})
        prices = obs.get("market", {}).get("prices", {})
        inventories = private.get("inventories", [])
        m_inv = obs.get("market", {}).get("inventory", {})

        # 1. Day 0 Kickstart: Rush hires and first animal
        if day == 0 and hour == 0:
            d0_hires = int(self.config.get("day0_hires", 5))
            for _ in range(d0_hires):
                market_orders.append(["HIRE"])
            d0_anim = str(self.config.get("day0_animal", "SHEEP"))
            cost = 500 if d0_anim == "SHEEP" else 400
            if money >= cost:
                market_orders.append(["BUY_ANIMAL", d0_anim, 1])
            return market_orders[:10]

        # 2. ANTI-SHOCK SELLING
        min_ratio = float(self.config.get("sell_price_ratio", 0.50))
        batch_limit = int(self.config.get("sell_batch_size", 8))

        # Products in order of value
        for item in ["WOOL", "MILK", "FERTILIZER", "MELON", "STRAWBERRY", "CARROT", "TOMATO", "EGG"]:
            qty = shed.get(item, 0)
            if qty <= 0:
                continue
            p = prices.get(item, 0)
            base_p = MARKET_PARAMS[item]["base"]
            cutoff_price = max(2, int(round(base_p * min_ratio)))
            if p < cutoff_price:
                # Do not sell below cutoff price before day 28!
                if day < 28:
                    continue

            cur_inv = m_inv.get(item)
            if cur_inv is None:
                cur_inv = estimate_inventory_from_price(item, p)

            batch = optimal_sale_batch(item, qty, cur_inv, min_marginal_ratio=min_ratio)
            actual_batch = min(qty, min(batch, batch_limit) if batch > 0 else (1 if p >= cutoff_price else 0))
            if actual_batch > 0:
                market_orders.append(["SELL", item, actual_batch])

        # Sell surplus wheat (keep small reserve)
        wheat_qty = shed.get("WHEAT", 0)
        cows_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "COW")
        sheep_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "SHEEP")
        total_animals = cows_now + sheep_now
        wheat_reserve = total_animals * 2
        if wheat_qty > wheat_reserve and len(market_orders) < 9:
            market_orders.append(["SELL", "WHEAT", min(10, wheat_qty - wheat_reserve)])

        # 3. HIRING
        hires_today = int(farm.get("hires_today", 0))
        target_hands = self.get_max_hands(day)
        min_cash_hand = int(self.config.get("min_cash_for_hand", 30))
        if hires_today < target_hands and money >= min_cash_hand:
            market_orders.append(["HIRE"])

        # 4. LAND EXPANSION
        if self.config.get("buy_land_enabled", True) and day <= 20:
            quads = len(farm.get("unlocked_quadrants", []))
            cost = next_land_cost(farm)
            if cost is not None and quads < 3:
                empty = count_empty_tiles(farm)
                d1 = int(self.config.get("expand_day_first", 7))
                d2 = int(self.config.get("expand_day_second", 13))
                reserve = int(self.config.get("land_reserve", 600))
                mult = float(self.config.get("buy_land_cash_multiplier", 1.10))
                thresh = cost * mult + reserve
                if quads == 1 and day >= d1 and money >= thresh and empty <= 15:
                    market_orders.append(["BUY_LAND"])
                elif quads == 2 and day >= d2 and money >= thresh and empty <= 28:
                    market_orders.append(["BUY_LAND"])

        # 5. BUY LIVESTOCK
        is_invest = day <= 22
        sheep_goal, cow_goal = self.get_dynamic_animal_goals(obs, farm)

        if is_invest:
            pasture_tiles = [
                (x, y, t) for y, r in enumerate(farm.get("tiles", []))
                for x, t in enumerate(r) if isinstance(t, dict) and t.get("kind") == "PASTURE"
            ]
            empty_pastures = [p for p in pasture_tiles if not p[2].get("animal")]
            carried_or_shed_sheep = shed.get("SHEEP", 0) + sum(inv.get("SHEEP", 0) for inv in inventories)
            carried_or_shed_cow = shed.get("COW", 0) + sum(inv.get("COW", 0) for inv in inventories)
            unplaced = carried_or_shed_sheep + carried_or_shed_cow

            vacant = max(0, len(empty_pastures) - unplaced)

            wool_p = prices.get("WOOL", 200)
            milk_p = prices.get("MILK", 160)

            # Buy Sheep if vacant and under sheep_goal
            if vacant > 0 and (sheep_now + carried_or_shed_sheep) < sheep_goal and money >= 600 and wool_p >= 40:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                vacant -= 1

            # Buy Cow if vacant and under cow_goal
            if vacant > 0 and (cows_now + carried_or_shed_cow) < cow_goal and money >= 500 and milk_p >= 40:
                market_orders.append(["BUY_ANIMAL", "COW", 1])

        # 6. BUY WHEAT FEED FOR ANIMALS
        wheat_avail = shed.get("WHEAT", 0) + sum(inv.get("WHEAT", 0) for inv in inventories)
        if total_animals > 0 and wheat_avail < total_animals and money >= 100:
            needed = min(10, (total_animals * 2) - wheat_avail)
            if needed > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", needed])

        # 7. BUY SEEDS FOR CROPS
        crop = self.choose_strategic_crop(obs, farm, day)
        seeds = private.get("seeds", {})
        if crop is not None and day < 28:
            seed_cnt = seeds.get(crop, 0)
            desired_seeds = 6 if len(farm.get("unlocked_quadrants", [])) > 1 else 4
            if seed_cnt < desired_seeds:
                cost = CROP_INFO[crop]["seed_cost"]
                if money >= (cost * 2 + 100):
                    market_orders.append(["BUY_SEED", crop, desired_seeds - seed_cnt])

        return market_orders[:10]

    def plan_units(self, obs: Dict[str, Any], farm: Dict[str, Any]) -> Tuple[List[Any], List[List[Any]]]:
        farmer_pos = tuple(farm["farmer"])
        hands = farm.get("hands", []) or []
        hand_positions = [tuple(p) for p in hands]
        private = obs.get("private", {})
        shed = private.get("shed", {})
        inventories = private.get("inventories", [])
        day = int(obs.get("day", 0))

        board_size = len(farm.get("tiles", []))
        half = board_size // 2
        shed_adjacent = [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]

        sheep_goal, cow_goal = self.get_dynamic_animal_goals(obs, farm)
        target_pastures = sheep_goal + cow_goal

        pastures = [
            (x, y, t) for y, r in enumerate(farm.get("tiles", []))
            for x, t in enumerate(r) if isinstance(t, dict) and t.get("kind") == "PASTURE"
        ]
        pasture_count = len(pastures)

        jobs: List[Dict[str, Any]] = []

        # 1. Pasture Animal Jobs: Feed, Harvest, Collect Fertilizer, Care
        for x, y, tile in pastures:
            animal = tile.get("animal")
            if animal in ANIMAL_INFO:
                if not tile.get("fed_today", False):
                    jobs.append({"priority": 0, "target": (x, y), "action": ["FEED"], "needs": "WHEAT"})
                if tile.get("yield_units", 0) > 0:
                    jobs.append({"priority": 1, "target": (x, y), "action": ["HARVEST"]})
                if tile.get("fertilizer_available", False):
                    jobs.append({"priority": 2, "target": (x, y), "action": ["COLLECT_FERTILIZER"]})
                if not tile.get("cared_today", False):
                    jobs.append({"priority": 3, "target": (x, y), "action": ["CARE"]})

        # 2. Wheat Pickup Jobs for animals that need feed
        held_wheat = sum(inv.get("WHEAT", 0) for inv in inventories)
        unfed_count = sum(1 for _, _, t in pastures if t.get("animal") and not t.get("fed_today", False))
        wheat_to_pick = min(shed.get("WHEAT", 0), max(0, unfed_count - held_wheat))
        for pos in shed_adjacent:
            if wheat_to_pick <= 0:
                break
            pick_amt = min(5, wheat_to_pick)
            jobs.append({"priority": 0, "target": pos, "action": ["PICKUP", "WHEAT", pick_amt]})
            wheat_to_pick -= pick_amt

        # 3. Animal Pickup and Placement
        empty_pastures = [(x, y) for x, y, t in pastures if not t.get("animal")]
        carried_animals = sum(inv.get("SHEEP", 0) + inv.get("COW", 0) for inv in inventories)
        vacant_slots = max(0, len(empty_pastures) - carried_animals)

        for species in ["SHEEP", "COW"]:
            stock = shed.get(species, 0)
            to_pick = min(stock, vacant_slots)
            for idx in range(to_pick):
                jobs.append({"priority": 1, "target": shed_adjacent[idx % 4], "action": ["PICKUP", species, 1]})
            vacant_slots -= to_pick

        # Animal Place jobs
        ep_idx = 0
        for species in ["SHEEP", "COW"]:
            carried = sum(inv.get(species, 0) for inv in inventories)
            while carried > 0 and ep_idx < len(empty_pastures):
                jobs.append({"priority": 1, "target": empty_pastures[ep_idx], "action": ["PLACE", species], "needs": species})
                ep_idx += 1
                carried -= 1

        # 4. Pasture Building Jobs (by hands or farmer)
        if day <= 22 and pasture_count < target_pastures:
            central_empties = get_unlocked_central_tiles(farm)
            needed_pastures = target_pastures - pasture_count
            for p_tile in central_empties[:needed_pastures]:
                jobs.append({"priority": 2, "target": p_tile, "action": ["BUILD_PASTURE"]})

        # 5. Assign Jobs to Units
        all_positions = [farmer_pos] + hand_positions
        all_inventories = inventories if len(inventories) >= len(all_positions) else [{} for _ in all_positions]
        actions: List[List[Any]] = [None] * len(all_positions)

        # First pass: drop produce if adjacent to shed and carrying items
        drop_thresh = int(self.config.get("drop_threshold", 2))
        for u_idx, (pos, inv) in enumerate(zip(all_positions, all_inventories)):
            if pos in shed_adjacent:
                carried_produce = sum(
                    inv.get(prod, 0) for prod in ["WOOL", "MILK", "FERTILIZER", "MELON", "STRAWBERRY", "CARROT"]
                )
                if carried_produce >= drop_thresh:
                    actions[u_idx] = ["DROP"]

        # Second pass: assign priority jobs
        for u_idx, (pos, inv) in enumerate(zip(all_positions, all_inventories)):
            if actions[u_idx] is not None:
                continue
            eligible = [j for j in jobs if not j.get("needs") or inv.get(j["needs"], 0) > 0]
            if eligible:
                best_job = min(eligible, key=lambda j: (j["priority"], distance(pos, j["target"])))
                jobs.remove(best_job)
                if pos == best_job["target"]:
                    actions[u_idx] = best_job["action"]
                else:
                    actions[u_idx] = next_step_toward(pos, best_job["target"])

        # Third pass: Farming Micro-Actions (Watering, Harvesting, Digging Weeds, Planting)
        crop = self.choose_strategic_crop(obs, farm, day)
        seeds = private.get("seeds", {})
        avail_seeds = seeds.get(crop, 0) if crop else 0
        weeds = find_weeds(farm)
        claimed_targets = set()

        for u_idx, (pos, inv) in enumerate(zip(all_positions, all_inventories)):
            if actions[u_idx] is not None:
                continue

            # Check standing tile: weed?
            x, y = pos
            tile = farm["tiles"][y][x]
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                actions[u_idx] = ["DIG"]
                continue

            # Urgent watering
            water_t = find_urgent_water_target(farm, pos)
            if water_t is not None and water_t not in claimed_targets:
                claimed_targets.add(water_t)
                if pos == water_t:
                    actions[u_idx] = ["WATER"]
                else:
                    actions[u_idx] = next_step_toward(pos, water_t)
                continue

            # Ready harvest
            harvest_t = find_harvest_target(farm, day, pos)
            if harvest_t is not None and harvest_t not in claimed_targets:
                claimed_targets.add(harvest_t)
                if pos == harvest_t:
                    actions[u_idx] = ["HARVEST"]
                else:
                    actions[u_idx] = next_step_toward(pos, harvest_t)
                continue

            # Dig weed nearby
            avail_w = [w for w in weeds if w not in claimed_targets]
            if avail_w:
                near_w = min(avail_w, key=lambda w: distance(pos, w))
                claimed_targets.add(near_w)
                if pos == near_w:
                    actions[u_idx] = ["DIG"]
                else:
                    actions[u_idx] = next_step_toward(pos, near_w)
                continue

            # Parallel Planting
            if crop is not None and avail_seeds > 0 and day < 28:
                empties = [p for p in unlocked_empty_tiles(farm) if p not in claimed_targets]
                if empties:
                    best_empty = min(empties, key=lambda p: distance(pos, p))
                    claimed_targets.add(best_empty)
                    if pos == best_empty:
                        actions[u_idx] = ["PLANT", crop]
                        avail_seeds -= 1
                    else:
                        actions[u_idx] = next_step_toward(pos, best_empty)
                    continue

            # Return to shed to drop if carrying items
            carried_items = sum(
                inv.get(prod, 0) for prod in ["WOOL", "MILK", "FERTILIZER", "MELON", "STRAWBERRY", "CARROT"]
            )
            if carried_items > 0:
                best_shed = min(shed_adjacent, key=lambda p: distance(pos, p))
                if pos in shed_adjacent:
                    actions[u_idx] = ["DROP"]
                else:
                    actions[u_idx] = next_step_toward(pos, best_shed)
                continue

            actions[u_idx] = ["PASS"]

        return actions[0], actions[1:]

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        player = obs["player"]
        farm = obs["farms"][player]

        # Final turn liquidation override
        observed_step = obs.get("step")
        if observed_step is not None and int(observed_step) >= FINAL_ACTION_STEP:
            market_orders, final_actions = final_turn_liquidation(obs, farm)
            return {
                "farmer": final_actions[0],
                "hands": final_actions[1:],
                "market": market_orders[:10],
            }

        market_orders = self.plan_market_orders(obs, farm)
        farmer_act, hand_acts = self.plan_units(obs, farm)

        return {
            "farmer": farmer_act,
            "hands": hand_acts,
            "market": market_orders[:10],
        }

