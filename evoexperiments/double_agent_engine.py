"""Double Agent Policy Engine for Kaggriculture - Grandmaster Center-Ring Edition.

Key Optimizations:
1. Central Piece Animal Ring:
   - Animals (Pastures & Coops) are strictly clustered in the inner central ring (x in [2..7], y in [2..7])
     immediately adjacent to the center shed (4.5, 4.5).
   - Minimizes walking latency to 1 turn for feeding, watering, fertilizer collection, and dropping to shed.
2. Outer Crop Prioritization:
   - All outer tiles (rings 0, 1, 8, 9) are 100% dedicated to high-margin cash crops (Melon, Strawberry, Carrot, Wheat).
3. Fast Expansion to Max 3 Quadrants:
   - Aggressively unlocks Quadrant 2 (NE) around Day 7 and Quadrant 3 (SW) around Day 11.
   - Strictly caps at 3 quadrants (75 tiles) to save the $4,000 late-game cost.
4. Town-Shop Dynamic Livestock Portfolio:
   - Scales Sheep (Yarn Store / Wool $200), Cows (Ice Cream / Pizza / Milk $160), and Geese (Bakery / Brunch / Egg $50).
   - All animals generate 1 fertilizer ($100 base) daily, collected and dropped to shed.
5. Scaled 11-Hand Workforce:
   - Expands to 11 hands as soon as Quadrant 3 unlocks (Days 11-28).
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
    unlocked_empty_tiles,
)

DEFAULT_GENOME: Dict[str, Any] = {
    # Livestock Goals (strictly 4 per species, max 12 total in central ring)
    "base_sheep_goal": 4,
    "base_cow_goal": 4,
    "base_goose_goal": 4,
    "day0_hires": 3,
    "day0_animal": "SHEEP",
    "day0_animal_qty": 4,
    "cash_reserve": 400,
    # Workforce Schedule
    "max_hands_p1": 4,   # Days 0-3
    "max_hands_p2": 7,   # Days 4-6
    "max_hands_p3": 8,   # Days 7-10
    "max_hands_p4": 11,  # Days 11-28 (Mass workforce)
    "min_cash_for_hand": 25,
    # Land Expansion (Max 3 Quadrants)
    "expand_day_first": 7,
    "expand_day_second": 11,
    "land_reserve": 450,
    "buy_land_cash_multiplier": 1.08,
    "buy_land_enabled": True,
    # Crops
    "max_wheat_tiles": 2,
    "melon_weight": 2.2,
    "strawberry_weight": 2.2,
    "carrot_weight": 0.9,
    # Market Trading
    "sell_price_ratio": 0.18,
    "sell_batch_size": 11,
    "drop_threshold": 2,
    "feed_reserve_mult": 1.25,
}

GENOME_SPECS: Dict[str, Dict[str, Any]] = {
    "base_sheep_goal": {"type": "int", "min": 3, "max": 5},
    "base_cow_goal": {"type": "int", "min": 3, "max": 5},
    "base_goose_goal": {"type": "int", "min": 2, "max": 5},
    "day0_hires": {"type": "int", "min": 3, "max": 4},
    "day0_animal_qty": {"type": "int", "min": 3, "max": 4},
    "cash_reserve": {"type": "int", "min": 300, "max": 550},
    "max_hands_p1": {"type": "int", "min": 3, "max": 5},
    "max_hands_p2": {"type": "int", "min": 6, "max": 8},
    "max_hands_p3": {"type": "int", "min": 7, "max": 9},
    "max_hands_p4": {"type": "int", "min": 10, "max": 12},
    "min_cash_for_hand": {"type": "int", "min": 15, "max": 35},
    "expand_day_first": {"type": "int", "min": 6, "max": 8},
    "expand_day_second": {"type": "int", "min": 10, "max": 12},
    "land_reserve": {"type": "int", "min": 350, "max": 600},
    "buy_land_cash_multiplier": {"type": "float", "min": 1.05, "max": 1.15},
    "max_wheat_tiles": {"type": "int", "min": 1, "max": 3},
    "melon_weight": {"type": "float", "min": 1.8, "max": 2.8},
    "strawberry_weight": {"type": "float", "min": 1.8, "max": 2.8},
    "carrot_weight": {"type": "float", "min": 0.6, "max": 1.3},
    "sell_price_ratio": {"type": "float", "min": 0.14, "max": 0.28},
    "sell_batch_size": {"type": "int", "min": 8, "max": 13},
    "drop_threshold": {"type": "int", "min": 1, "max": 3},
    "feed_reserve_mult": {"type": "float", "min": 1.1, "max": 1.5},
}


def get_town_drain(unlocked_shops: List[str]) -> Dict[str, float]:
    rates = {item: 1.0 for item in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]}
    for shop in unlocked_shops:
        demands = SHOP_DEMAND.get(shop, {})
        for item, qty in demands.items():
            rates[item] = rates.get(item, 0.0) + (qty * 6.0)
    return rates


def is_central_ring_tile(pos: Tuple[int, int]) -> bool:
    """Return True if tile is inside the inner ring surrounding the shed (x,y in [2..7])."""
    x, y = pos
    return 2 <= x <= 7 and 2 <= y <= 7


def get_central_animal_tiles(farm: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return empty unlocked tiles inside the central animal ring, sorted by distance to shed center (4.5, 4.5)."""
    empty = unlocked_empty_tiles(farm)
    # Strictly prefer central ring tiles first
    central = [p for p in empty if is_central_ring_tile(p)]
    central.sort(key=lambda p: (abs(p[0] - 4.5) + abs(p[1] - 4.5), distance((4, 4), p)))
    if central:
        return central
    # Fallback to closest available
    empty.sort(key=lambda p: (abs(p[0] - 4.5) + abs(p[1] - 4.5)))
    return empty


def get_outer_crop_tiles(farm: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return empty unlocked tiles outside the central ring, perfect for crops."""
    empty = unlocked_empty_tiles(farm)
    outer = [p for p in empty if not is_central_ring_tile(p)]
    # Outer tiles sorted from outside inward
    outer.sort(key=lambda p: -(abs(p[0] - 4.5) + abs(p[1] - 4.5)))
    return outer


class DoubleAgentPolicy:
    """Grandmaster Double Agent Policy Engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(DEFAULT_GENOME)
        if config:
            self.config.update(config)

    def get_max_hands(self, day: int) -> int:
        if day < 4:
            return int(self.config.get("max_hands_p1", 4))
        elif day < 7:
            return int(self.config.get("max_hands_p2", 7))
        elif day < 11:
            return int(self.config.get("max_hands_p3", 8))
        elif day < 28:
            return int(self.config.get("max_hands_p4", 11))
        else:
            return 7

    def get_dynamic_goals(self, obs: Dict[str, Any], farm: Dict[str, Any]) -> Tuple[int, int, int]:
        """Strictly cap animals at 4 per species (max 12 total), perfectly fitting the central ring."""
        unlocked_shops = obs.get("town", {}).get("unlocked_shops", [])
        drain = get_town_drain(unlocked_shops)

        # 4 Sheep from Day 0
        target_sheep = 4
        # 4 Cows from Day 7
        target_cow = 4
        # 4 Geese from Day 11 if egg demand exists, else 2
        has_bakery = drain.get("EGG", 1.0) > 1.2
        target_goose = 4 if has_bakery else 2

        return target_sheep, target_cow, target_goose

    def choose_strategic_crop(self, obs: Dict[str, Any], farm: Dict[str, Any], day: int) -> Optional[str]:
        if day >= 28:
            return None
        if day >= 26:
            return "WHEAT"
        if day >= 24:
            return "CARROT"

        # On Days 0 to 2: Prioritize MELON!
        # Melon takes 10 days to mature and yields 6 melons with bonus watering.
        # This creates the Day 10-11 liquidity surge that funds Land 2 and the massive 11-hand workforce.
        melon_cnt = count_crop(farm, "MELON")
        if day <= 2 and melon_cnt < 14:
            return "MELON"

        money = farm.get("money", 0)
        current_wheat = count_crop(farm, "WHEAT")
        max_wheat = int(self.config.get("max_wheat_tiles", 2))

        if money < 300 and current_wheat < max_wheat:
            return "WHEAT"

        prices = obs.get("market", {}).get("prices", {})
        melon_p = prices.get("MELON", 250)
        straw_p = prices.get("STRAWBERRY", 120)
        carrot_p = prices.get("CARROT", 35)

        m_wt = float(self.config.get("melon_weight", 2.2))
        s_wt = float(self.config.get("strawberry_weight", 2.2))
        c_wt = float(self.config.get("carrot_weight", 0.9))

        melon_score = (melon_p / 250.0) * m_wt
        straw_score = (straw_p / 120.0) * s_wt

        straw_cnt = count_crop(farm, "STRAWBERRY")

        if day <= 16:
            if straw_p >= 45 and straw_cnt < 35:
                return "STRAWBERRY"
            if melon_score >= straw_score and melon_cnt < 18 and melon_p >= 120:
                return "MELON"
            if melon_cnt < 14 and melon_p >= 100:
                return "MELON"
            return "CARROT"
        elif day <= 22:
            if straw_p >= 50 and straw_cnt < 32:
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

        # 1. Day 0 Grandmaster Opening: 3 Hires, 4 Sheep, 4 Wheat, 10 Melon Seeds, 9 Wheat Seeds
        if day == 0 and hour <= 1 and money >= 2900:
            return [
                ["HIRE"], ["HIRE"], ["HIRE"],
                ["BUY_ANIMAL", "SHEEP", 4],
                ["BUY_PRODUCT", "WHEAT", 4],
                ["BUY_SEED", "MELON", 10],
                ["BUY_SEED", "WHEAT", 9],
            ]

        # 2. ANTI-SHOCK SELLING
        min_ratio = float(self.config.get("sell_price_ratio", 0.18))
        batch_limit = int(self.config.get("sell_batch_size", 11))
        unlocked_shops = obs.get("town", {}).get("unlocked_shops", [])
        drain_rates = get_town_drain(unlocked_shops)

        for item in ["WOOL", "MILK", "FERTILIZER", "MELON", "STRAWBERRY", "CARROT", "TOMATO", "EGG"]:
            qty = shed.get(item, 0)
            if qty <= 0:
                continue
            p = prices.get(item, 0)
            base_p = MARKET_PARAMS[item]["base"]
            cutoff_price = max(2, int(round(base_p * min_ratio)))
            has_drain = drain_rates.get(item, 1.0) > 1.5

            cur_inv = m_inv.get(item)
            if cur_inv is None:
                cur_inv = estimate_inventory_from_price(item, p)

            if has_drain:
                batch = optimal_sale_batch(item, qty, cur_inv, min_marginal_ratio=min_ratio)
                actual_batch = min(qty, min(batch, batch_limit) if batch > 0 else (1 if p >= cutoff_price else 0))
            else:
                # No town drain: supply does not recover naturally. Sell steadily!
                actual_batch = min(qty, min(8, batch_limit))

            if actual_batch > 0:
                market_orders.append(["SELL", item, actual_batch])

        # Strict feed protection: never sell wheat below reserve
        wheat_qty = shed.get("WHEAT", 0)
        cows_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "COW")
        sheep_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "SHEEP")
        geese_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "GOOSE")
        total_animals = cows_now + sheep_now + geese_now

        feed_mult = float(self.config.get("feed_reserve_mult", 1.25))
        wheat_reserve = int(total_animals * feed_mult)
        if wheat_qty > (wheat_reserve + 8) and len(market_orders) < 9:
            market_orders.append(["SELL", "WHEAT", min(10, wheat_qty - wheat_reserve)])

        # 3. ABSOLUTE FEED SECURITY: If animals are low on wheat, BUY FEED FIRST!
        wheat_avail = shed.get("WHEAT", 0) + sum(inv.get("WHEAT", 0) for inv in inventories)
        if total_animals > 0 and wheat_avail < total_animals:
            needed = min(10, total_animals - wheat_avail)
            if needed > 0 and money >= 25:
                market_orders.append(["BUY_PRODUCT", "WHEAT", needed])

        # 4. HIRING (Dynamic Fibonacci Cost)
        hires_today = int(farm.get("hires_today", 0))
        target_hands = self.get_max_hands(day)
        FIB_COSTS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        hire_cost = FIB_COSTS[hires_today] if hires_today < len(FIB_COSTS) else 144
        if hires_today < target_hands and money >= hire_cost:
            if day > 0 and day < 4 and total_animals > 0 and wheat_avail < total_animals and money < (hire_cost + 25):
                pass
            else:
                market_orders.append(["HIRE"])

        # 5. FAST LAND EXPANSION TO MAX 3 QUADRANTS
        quads = len(farm.get("unlocked_quadrants", []))
        if self.config.get("buy_land_enabled", True) and day <= 22:
            cost = next_land_cost(farm)
            # Richardwe rule: Max 3 quadrants (NW, NE, SW). Never buy 4th quadrant!
            if cost is not None and quads < 3:
                d1 = int(self.config.get("expand_day_first", 7))
                d2 = int(self.config.get("expand_day_second", 11))
                reserve = int(self.config.get("land_reserve", 450))
                mult = float(self.config.get("buy_land_cash_multiplier", 1.08))
                thresh = cost * mult + reserve
                if quads == 1 and day >= d1 and money >= thresh:
                    market_orders.append(["BUY_LAND"])
                elif quads == 2 and day >= d2 and money >= thresh:
                    market_orders.append(["BUY_LAND"])

        # 6. BUY LIVESTOCK
        # Before Day 7: We already have 4 Sheep from Day 0. Let Melons and Wheat occupy the outer tiles!
        # Day 7 to 10 (Land 1 NE unlocked): Buy 4 Cows (or Sheep) if money allows
        # Day 11+ (Melon surge + Land 2 SW unlocked): Buy Geese / Cows / Sheep up to dynamic goals
        is_invest = day <= 22
        sheep_goal, cow_goal, goose_goal = self.get_dynamic_goals(obs, farm)
        cash_res = int(self.config.get("cash_reserve", 400))
        wool_p = prices.get("WOOL", 200)
        milk_p = prices.get("MILK", 160)
        avail_money = money - cash_res

        if is_invest and day >= 7 and avail_money >= 300:
            central_empties = get_central_animal_tiles(farm)
            unplaced_anim_count = (
                shed.get("SHEEP", 0) + shed.get("COW", 0) + shed.get("GOOSE", 0) +
                sum(sum(inv.get(a, 0) for a in ["SHEEP", "COW", "GOOSE"]) for inv in inventories)
            )
            empty_struct_count = len([
                t for r in farm.get("tiles", []) for t in r
                if isinstance(t, dict) and t.get("kind") in ("PASTURE", "COOP") and not t.get("animal")
            ])
            animal_slots_available = len(central_empties) + empty_struct_count - unplaced_anim_count

            carried_or_shed_sheep = shed.get("SHEEP", 0) + sum(inv.get("SHEEP", 0) for inv in inventories)
            carried_or_shed_cow = shed.get("COW", 0) + sum(inv.get("COW", 0) for inv in inventories)
            carried_or_shed_goose = shed.get("GOOSE", 0) + sum(inv.get("GOOSE", 0) for inv in inventories)

            if animal_slots_available > 0:
                if quads >= 2 and day < 11:
                    if (cows_now + carried_or_shed_cow) < 4 and avail_money >= 400:
                        market_orders.append(["BUY_ANIMAL", "COW", 1])
                elif quads >= 3 and day >= 11:
                    if (geese_now + carried_or_shed_goose) < goose_goal and avail_money >= 300:
                        market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
                    elif (sheep_now + carried_or_shed_sheep) < sheep_goal and avail_money >= 500 and wool_p >= 35:
                        market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                    elif (cows_now + carried_or_shed_cow) < cow_goal and avail_money >= 400 and milk_p >= 35:
                        market_orders.append(["BUY_ANIMAL", "COW", 1])

        # 7. BUY SEEDS FOR CROPS (Mass Outer Strawberry Farm)
        crop = self.choose_strategic_crop(obs, farm, day)
        seeds = private.get("seeds", {})
        straw_cnt = count_crop(farm, "STRAWBERRY") + seeds.get("STRAWBERRY", 0)

        # Priority 1: On Days 3-18, scale Strawberries up to 32 tiles!
        if day >= 3 and day <= 18 and straw_cnt < 32 and money >= 280 and len(market_orders) < 9:
            to_buy = min(6, 32 - straw_cnt)
            if to_buy > 0:
                market_orders.append(["BUY_SEED", "STRAWBERRY", to_buy])
        # Priority 2: Strategic crop seeds (Melon / Carrot / Wheat)
        elif crop is not None and day < 28 and len(market_orders) < 9:
            seed_cnt = seeds.get(crop, 0)
            desired_seeds = 8 if quads > 1 else 4
            if seed_cnt < desired_seeds:
                cost = CROP_INFO[crop]["seed_cost"]
                if money >= (cost * 2 + 80):
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

        structures = [
            (x, y, t) for y, r in enumerate(farm.get("tiles", []))
            for x, t in enumerate(r) if isinstance(t, dict) and t.get("kind") in ("PASTURE", "COOP")
        ]
        pastures = [s for s in structures if s[2].get("kind") == "PASTURE"]
        coops = [s for s in structures if s[2].get("kind") == "COOP"]

        jobs: List[Dict[str, Any]] = []

        # 1. Animal Maintenance: Feed, Harvest, Fertilizer Collection, Care
        for x, y, tile in structures:
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

        # 2. Wheat Pickup Jobs for animals needing feed
        held_wheat = sum(inv.get("WHEAT", 0) for inv in inventories)
        unfed_count = sum(1 for _, _, t in structures if t.get("animal") and not t.get("fed_today", False))
        wheat_to_pick = min(shed.get("WHEAT", 0), max(0, unfed_count - held_wheat))
        for pos in shed_adjacent:
            if wheat_to_pick <= 0:
                break
            pick_amt = min(5, wheat_to_pick)
            jobs.append({"priority": 0, "target": pos, "action": ["PICKUP", "WHEAT", pick_amt]})
            wheat_to_pick -= pick_amt

        # 3. Animal Pickup and Placement
        empty_pastures = [(x, y) for x, y, t in pastures if not t.get("animal")]
        empty_coops = [(x, y) for x, y, t in coops if not t.get("animal")]

        carried_pasture = sum(inv.get("SHEEP", 0) + inv.get("COW", 0) for inv in inventories)
        carried_coop = sum(inv.get("GOOSE", 0) for inv in inventories)

        vacant_pasture_slots = max(0, len(empty_pastures) - carried_pasture)
        vacant_coop_slots = max(0, len(empty_coops) - carried_coop)

        for species in ["SHEEP", "COW"]:
            stock = shed.get(species, 0)
            to_pick = min(stock, vacant_pasture_slots)
            for idx in range(to_pick):
                jobs.append({"priority": 1, "target": shed_adjacent[idx % len(shed_adjacent)], "action": ["PICKUP", species, 1]})
            vacant_pasture_slots -= to_pick

        stock_goose = shed.get("GOOSE", 0)
        to_pick_goose = min(stock_goose, vacant_coop_slots)
        for idx in range(to_pick_goose):
            jobs.append({"priority": 1, "target": shed_adjacent[idx % len(shed_adjacent)], "action": ["PICKUP", "GOOSE", 1]})

        ep_idx = 0
        for species in ["SHEEP", "COW"]:
            carried = sum(inv.get(species, 0) for inv in inventories)
            while carried > 0 and ep_idx < len(empty_pastures):
                jobs.append({"priority": 1, "target": empty_pastures[ep_idx], "action": ["PLACE", species], "needs": species})
                ep_idx += 1
                carried -= 1

        ec_idx = 0
        carried_g = sum(inv.get("GOOSE", 0) for inv in inventories)
        while carried_g > 0 and ec_idx < len(empty_coops):
            jobs.append({"priority": 1, "target": empty_coops[ec_idx], "action": ["PLACE", "GOOSE"], "needs": "GOOSE"})
            ec_idx += 1
            carried_g -= 1

        # 4. Central Ring Structure Construction
        # ONLY build a pasture or coop if an animal is actually waiting in shed/inventory!
        if day <= 22:
            central_empties = get_central_animal_tiles(farm)
            unplaced_pasture_animals = (
                shed.get("SHEEP", 0) + shed.get("COW", 0) +
                sum(inv.get("SHEEP", 0) + inv.get("COW", 0) for inv in inventories)
            )
            needed_pastures = max(0, unplaced_pasture_animals - len(empty_pastures))

            unplaced_coop_animals = (
                shed.get("GOOSE", 0) +
                sum(inv.get("GOOSE", 0) for inv in inventories)
            )
            needed_coops = max(0, unplaced_coop_animals - len(empty_coops))

            c_idx = 0
            while needed_pastures > 0 and c_idx < len(central_empties):
                jobs.append({"priority": 1, "target": central_empties[c_idx], "action": ["BUILD_PASTURE"]})
                c_idx += 1
                needed_pastures -= 1

            while needed_coops > 0 and c_idx < len(central_empties):
                jobs.append({"priority": 1, "target": central_empties[c_idx], "action": ["BUILD_COOP"]})
                c_idx += 1
                needed_coops -= 1

        # 5. Assign Jobs to Units
        all_positions = [farmer_pos] + hand_positions
        all_inventories = [inventories[i] if i < len(inventories) else {} for i in range(len(all_positions))]
        actions: List[List[Any]] = [None] * len(all_positions)

        # First pass: drop produce if adjacent to shed and carrying items
        drop_thresh = int(self.config.get("drop_threshold", 2))
        for u_idx, (pos, inv) in enumerate(zip(all_positions, all_inventories)):
            if pos in shed_adjacent:
                carried_produce = sum(
                    inv.get(prod, 0) for prod in ["WOOL", "MILK", "FERTILIZER", "MELON", "STRAWBERRY", "CARROT", "EGG"]
                )
                if carried_produce >= drop_thresh:
                    actions[u_idx] = ["DROP"]

        # Second pass: assign priority maintenance & placement jobs
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

        # Third pass: Farming Micro-Actions (Outer Tiles Crop Farming, Weeds, Priority Watering, Harvesting)
        seeds = dict(private.get("seeds", {}))
        weeds = find_weeds(farm)
        claimed_targets = set()
        unplaced_anim_count = (
            shed.get("SHEEP", 0) + shed.get("COW", 0) + shed.get("GOOSE", 0) +
            sum(sum(inv.get(a, 0) for a in ["SHEEP", "COW", "GOOSE"]) for inv in inventories)
        )
        animal_slots_available = max(0, 12 - (len(pastures) + len(coops) + unplaced_anim_count))

        for u_idx, (pos, inv) in enumerate(zip(all_positions, all_inventories)):
            if actions[u_idx] is not None:
                continue

            x, y = pos
            tile = farm["tiles"][y][x]

            # Choose seed to plant
            plant_seed = None
            if day <= 16 and seeds.get("MELON", 0) > 0:
                plant_seed = "MELON"
            elif day <= 22 and seeds.get("STRAWBERRY", 0) > 0:
                plant_seed = "STRAWBERRY"
            elif day <= 20 and seeds.get("TOMATO", 0) > 0:
                plant_seed = "TOMATO"
            elif day <= 26 and seeds.get("CARROT", 0) > 0:
                plant_seed = "CARROT"
            elif seeds.get("WHEAT", 0) > 0:
                plant_seed = "WHEAT"

            # 1. IMMEDIATE LOCAL ACTIONS ON CURRENT TILE (0 distance, instant execution!)
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                actions[u_idx] = ["DIG"]
                continue

            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop_age = day - tile.get("planted_day", day)
                crop_name = tile.get("crop")
                first_yield = CROP_INFO.get(crop_name, {}).get("first_yield_day", 999)
                if tile.get("yield_units", 0) > 0 and crop_age >= first_yield:
                    actions[u_idx] = ["HARVEST"]
                    continue
                if inv.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < day:
                    if (crop_name == "MELON" and day >= 5) or crop_name == "STRAWBERRY":
                        actions[u_idx] = ["FERTILIZE"]
                        continue
                if not tile.get("watered_today", False):
                    actions[u_idx] = ["WATER"]
                    continue

            if tile is None and plant_seed is not None and day < 28:
                if not is_central_ring_tile(pos) or animal_slots_available <= 0:
                    actions[u_idx] = ["PLANT", plant_seed]
                    seeds[plant_seed] -= 1
                    continue

            # 2. MOVEMENT TO TARGETS (Balanced Distance Matching)
            best_planting_target = None
            dist_plant = 999
            if plant_seed is not None and day < 28:
                outer_empties = [p for p in get_outer_crop_tiles(farm) if p not in claimed_targets]
                all_empties = [p for p in unlocked_empty_tiles(farm) if p not in claimed_targets]
                candidate_empties = outer_empties if outer_empties else all_empties
                if candidate_empties:
                    best_planting_target = min(candidate_empties, key=lambda p: distance(pos, p))
                    dist_plant = distance(pos, best_planting_target)

            best_water_target = None
            dist_water = 999
            is_emergency_water = False
            unwatered = []
            for px, py, pcrop in plant_tiles(farm):
                if (px, py) in claimed_targets:
                    continue
                ptile = farm["tiles"][py][px]
                if not ptile.get("watered_today", False):
                    consec = ptile.get("consecutive_unwatered", 0)
                    if consec >= 1:
                        p_score = -10  # Emergency: plant dies tonight if not watered!
                    elif pcrop in ("MELON", "STRAWBERRY"):
                        p_score = 0
                    elif pcrop in ("CARROT", "TOMATO"):
                        p_score = 1
                    else:
                        p_score = 2
                    unwatered.append((p_score, distance(pos, (px, py)), px, py))
            if unwatered:
                unwatered.sort()
                p_s, d_w, wx, wy = unwatered[0]
                best_water_target = (wx, wy)
                dist_water = d_w
                is_emergency_water = (p_s == -10)

            best_harvest_target = None
            dist_harvest = 999
            harvest_t = find_harvest_target(farm, day, pos)
            if harvest_t is not None and harvest_t not in claimed_targets:
                best_harvest_target = harvest_t
                dist_harvest = distance(pos, harvest_t)

            # Choose the closest high-priority action:
            # 1. Emergency watering (prevents plant death and weed explosion)
            if best_water_target is not None and is_emergency_water:
                claimed_targets.add(best_water_target)
                actions[u_idx] = next_step_toward(pos, best_water_target)
                continue

            # 2. Immediate harvest
            if best_harvest_target is not None and dist_harvest <= 3:
                claimed_targets.add(best_harvest_target)
                actions[u_idx] = next_step_toward(pos, best_harvest_target)
                continue

            # 3. Regular watering (keep existing investments alive)
            if best_water_target is not None and (dist_water <= 4 or best_planting_target is None):
                claimed_targets.add(best_water_target)
                actions[u_idx] = next_step_toward(pos, best_water_target)
                continue

            # 4. Planting seeds
            if best_planting_target is not None:
                claimed_targets.add(best_planting_target)
                actions[u_idx] = next_step_toward(pos, best_planting_target)
                continue

            if best_water_target is not None:
                claimed_targets.add(best_water_target)
                actions[u_idx] = next_step_toward(pos, best_water_target)
                continue

            avail_w = [w for w in weeds if w not in claimed_targets]
            if avail_w:
                near_w = min(avail_w, key=lambda w: distance(pos, w))
                # Only walk to clear weeds if nearby or inside central ring
                if is_central_ring_tile(near_w) or distance(pos, near_w) <= 3:
                    claimed_targets.add(near_w)
                    actions[u_idx] = next_step_toward(pos, near_w)
                    continue

            carried_items = sum(
                inv.get(prod, 0) for prod in ["WOOL", "MILK", "FERTILIZER", "MELON", "STRAWBERRY", "CARROT", "EGG"]
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

    def __call__(self, obs: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        player = obs["player"]
        farm = obs["farms"][player]

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

