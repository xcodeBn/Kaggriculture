"""Enhanced Kaggriculture Strategy Agent.

Key improvements:
1. Weed Clearance: Hands & farmer actively DIG weeds on unlocked tiles.
2. Livestock Placement:
   - Farmer builds pastures until structure_count >= sum(goals) (fixes shed stock bug).
   - High priority placement (Priority 1) for hands, plus farmer placement support.
3. Feed Security:
   - Ensures wheat planting scales with herd size (0.5 * animal_count).
   - Automatically maintains wheat seed buffer.
4. Investment Phasing:
   - Days 0-22: Aggressive expansion (BUY_LAND, livestock purchases).
   - Days 23-30: Freeze land & livestock buys; focus on production, staggered sales & liquidation.
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

import importlib.util
_spec = importlib.util.spec_from_file_location("market_model", str(ROOT / "evoexperiments" / "market_model.py"))
market_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(market_model)

calculate_price = market_model.calculate_price
optimal_sale_batch = market_model.optimal_sale_batch
estimate_inventory_from_price = market_model.estimate_inventory_from_price

import submission
from submission import (
    ANIMAL_INFO,
    CROP_INFO,
    CROP_ORDER,
    FINAL_ACTION_STEP,
    LAND_COSTS,
    SHOP_DEMAND,
    animal_goals,
    count_crop,
    crop_demand_score,
    crop_ratio,
    distance,
    find_harvest_target,
    find_nearest_plant,
    find_urgent_water_target,
    next_land_cost,
    next_step_toward,
    plant_tiles,
    set_policy_config,
    unlocked_empty_tiles,
)

ENHANCED_CONFIG: Dict[str, Any] = {
    "buy_land_enabled": True,
    "land_reserve": 250,
    "buy_land_cash_multiplier": 1.15,
    "cash_reserve": 300,
    "max_animals": 12,
    "cow_goal": 6,
    "sheep_goal": 6,
    "max_hands": 8,
    "min_cash_for_hand": 50,
    "sell_batch_size": 10,
    "sell_price_ratio": 0.30,
    "sell_min_price": 1,
    "town_demand_weight": 1.0,
    "town_price_weight": 0.3,
    "town_scarcity_weight": 0.097,
    "wheat_ratio": 1.0,
    "carrot_ratio": 1.25,
    "tomato_ratio": 1.0,
    "strawberry_ratio": 1.05,
    "melon_ratio": 0.57,
    "hire_enabled": True,
}


def find_weeds(farm: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return coordinates of all weeds on unlocked tiles."""
    weeds = []
    for y, row in enumerate(farm.get("tiles", [])):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                weeds.append((x, y))
    return weeds


def choose_enhanced_crop(obs: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Choose crop with feed security priority."""
    player = obs["player"]
    farm = obs["farms"][player]
    goals = animal_goals(config)
    total_animals = sum(goals.values())

    # Feed security: ensure enough wheat tiles are active
    if total_animals > 0:
        current_wheat = count_crop(farm, "WHEAT")
        target_wheat = max(3, math.ceil(total_animals * 0.4))
        shed_wheat = int(obs["private"].get("shed", {}).get("WHEAT", 0))
        if current_wheat < target_wheat or shed_wheat < (total_animals * 2):
            return "WHEAT"


    # Standard profitability proxy
    market = obs.get("market", {})
    prices = market.get("prices", {})
    scores: Dict[str, float] = {}

    for crop in CROP_ORDER:
        info = CROP_INFO[crop]
        base = info["base_price"]
        market_ratio = prices.get(crop, base) / base
        profit_proxy = max(
            0.0,
            (prices.get(crop, base) - info["seed_cost"]) / max(info["first_yield_day"], 1),
        )
        town = crop_demand_score(obs, crop)
        profit_component = min(2.0, profit_proxy / 30.0)
        scores[crop] = (
            town + 0.25 * market_ratio + 0.20 * profit_component
        ) * crop_ratio(crop)

    return max(scores, key=scores.get)


def enhanced_animal_hand_actions(
    obs: Dict[str, Any],
    farm: Dict[str, Any],
    config: Dict[str, Any],
) -> List[Optional[List[Any]]]:
    """Assign animal tasks with HIGH-PRIORITY placement to prevent shed stranding."""
    goals = animal_goals(config)
    hands = farm.get("hands", []) or []
    if not goals:
        return [None for _ in hands]

    private = obs.get("private", {}) or {}
    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", []) or []
    hand_inventories = [
        inventories[i + 1] if i + 1 < len(inventories) else {}
        for i in range(len(hands))
    ]
    positions = [tuple(p) for p in hands]

    jobs: List[Dict[str, Any]] = []
    animals: List[Tuple[int, int, Dict[str, Any]]] = []
    empty_structures: Dict[str, List[Tuple[int, int]]] = {}

    for y, row in enumerate(farm.get("tiles", [])):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            species = tile.get("animal")
            if species in goals:
                animals.append((x, y, tile))
                if not tile.get("fed_today", False):
                    jobs.append({"priority": 0, "target": (x, y), "action": ["FEED"], "needs": "WHEAT"})
                if tile.get("yield_units", 0) > 0:
                    jobs.append({"priority": 2, "target": (x, y), "action": ["HARVEST"]})
                if tile.get("fertilizer_available", False):
                    jobs.append({"priority": 3, "target": (x, y), "action": ["COLLECT_FERTILIZER"]})
                if not tile.get("cared_today", False):
                    jobs.append({"priority": 4, "target": (x, y), "action": ["CARE"]})
            elif tile.get("kind") in {ANIMAL_INFO[s]["structure"] for s in goals} and not tile.get("animal"):
                empty_structures.setdefault(tile["kind"], []).append((x, y))

    board_size = len(farm.get("tiles", []))
    half = board_size // 2
    shed_positions = [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]

    # Wheat pickup for hungry animals
    held_wheat = sum(int(inv.get("WHEAT", 0)) for inv in hand_inventories)
    wheat_needed = sum(not tile.get("fed_today", False) for _, _, tile in animals)
    wheat_to_pick = min(int(shed.get("WHEAT", 0)), max(0, wheat_needed - held_wheat))
    for position in shed_positions:
        if wheat_to_pick <= 0:
            break
        amount = min(5, wheat_to_pick)
        jobs.append({"priority": 0, "target": position, "action": ["PICKUP", "WHEAT", amount]})
        wheat_to_pick -= amount

    # High-Priority Animal Placement: Priority 1 (right after FEED)
    carried_animals = sum(int(inv.get(species, 0)) for inv in inventories for species in goals)
    available_places = sum(len(tiles) for tiles in empty_structures.values())
    stock_left_to_pick = max(0, available_places - carried_animals)
    place_slots = [p for structure in empty_structures.values() for p in structure]

    for species in goals:
        stock = int(shed.get(species, 0))
        animals_to_pick = min(stock, stock_left_to_pick)
        for idx in range(animals_to_pick):
            jobs.append({
                "priority": 1,
                "target": shed_positions[idx % len(shed_positions)],
                "action": ["PICKUP", species, 1],
            })
        stock_left_to_pick -= animals_to_pick

    slot_idx = 0
    for species in goals:
        carried = sum(int(inv.get(species, 0)) for inv in inventories)
        while carried > 0 and slot_idx < len(place_slots):
            pos = place_slots[slot_idx]
            jobs.append({
                "priority": 1,
                "target": pos,
                "action": ["PLACE", species],
                "needs": species,
            })
            slot_idx += 1
            carried -= 1

    assigned: List[Optional[List[Any]]] = []
    for pos, inv in zip(positions, hand_inventories):
        eligible = [j for j in jobs if not j.get("needs") or inv.get(j["needs"], 0) > 0]
        if not eligible:
            assigned.append(None)
            continue
        job = min(eligible, key=lambda item: (item["priority"], distance(pos, item["target"])))
        jobs.remove(job)
        assigned.append(job["action"] if pos == job["target"] else next_step_toward(pos, job["target"]))

    return assigned


class EnhancedAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(ENHANCED_CONFIG)
        if config:
            self.config.update(config)

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        set_policy_config(self.config)
        step = int(obs.get("step", 0))
        day = int(obs.get("day", 0))
        player = int(obs["player"])
        farm = obs["farms"][player]
        private = obs["private"]
        money = farm.get("money", 0)
        fx, fy = farm["farmer"]
        farmer_pos = (fx, fy)

        market_orders: List[List[Any]] = []

        # ----------------------------------------------------
        # 1. Market Orders: Land & Livestock Phase Control
        # ----------------------------------------------------
        # Cut off capital investments after Day 22 to guarantee ROI
        is_investment_phase = day <= 22

        # A. Buy Land (Investment phase only)
        if is_investment_phase and self.config["buy_land_enabled"]:
            cost = next_land_cost(farm)
            if cost is not None:
                empty = len(unlocked_empty_tiles(farm))
                threshold = max(
                    self.config["land_reserve"],
                    cost * self.config["buy_land_cash_multiplier"],
                )
                if money >= threshold and empty <= 2:
                    market_orders.append(["BUY_LAND"])

        # B. Hire Hands
        if self.config["hire_enabled"]:
            hires = int(farm.get("hires_today", 0))
            if hires < self.config["max_hands"] and money >= self.config["min_cash_for_hand"]:
                # Don't hire on turn 719
                if step < FINAL_ACTION_STEP:
                    market_orders.append(["HIRE"])

        goals = animal_goals(self.config)

        # C. Crop / Seed Purchases
        crop = choose_enhanced_crop(obs, self.config)
        seeds = private.get("seeds", {})
        desired_seeds = 3
        seed_count = seeds.get(crop, 0)
        if seed_count < desired_seeds:
            cost = CROP_INFO[crop]["seed_cost"]
            avail_cash = money - self.config["cash_reserve"]
            affordable = max(0, int(avail_cash // cost))
            buy_count = min(desired_seeds - seed_count, affordable, 3)
            if buy_count > 0:
                market_orders.append(["BUY_SEED", crop, buy_count])

        # Also buy wheat seeds if feed security needs them
        if goals and crop != "WHEAT":
            wheat_seeds = seeds.get("WHEAT", 0)
            if wheat_seeds < 3 and money - self.config["cash_reserve"] >= 30:
                market_orders.append(["BUY_SEED", "WHEAT", 3 - wheat_seeds])

        # D. Livestock Purchases (Investment phase only)
        if goals and is_investment_phase:
            shed = private.get("shed", {})
            inventories = private.get("inventories", [])
            structures = {ANIMAL_INFO[s]["structure"] for s in goals}
            empty_structures = sum(
                1 for row in farm.get("tiles", []) for tile in row
                if isinstance(tile, dict) and tile.get("kind") in structures and not tile.get("animal")
            )
            occupied_by_species = {
                s: sum(1 for row in farm.get("tiles", []) for tile in row
                       if isinstance(tile, dict) and tile.get("animal") == s)
                for s in goals
            }
            stock_by_species = {
                s: int(shed.get(s, 0)) + sum(int(inv.get(s, 0)) for inv in inventories)
                for s in goals
            }
            stock_total = sum(stock_by_species.values())
            # ONLY buy if we have empty structures ready!
            vacant_for_stock = max(0, empty_structures - stock_total)
            species_order = [s for s in ("COW", "SHEEP", "GOOSE") if s in goals]
            for s in goals:
                if s not in species_order:
                    species_order.append(s)

            for animal_species in species_order:
                deficit = goals[animal_species] - occupied_by_species[animal_species] - stock_by_species[animal_species]
                animal_price = ANIMAL_INFO[animal_species]["cost"]
                if deficit > 0 and vacant_for_stock > 0 and money - self.config["cash_reserve"] >= animal_price:
                    market_orders.append(["BUY_ANIMAL", animal_species, 1])
                    break

        # Emergency wheat buy if animals might starve - ACTIVE ON ALL DAYS!
        if goals:
            shed = private.get("shed", {})
            inventories = private.get("inventories", [])
            occupied_by_species = {
                s: sum(1 for row in farm.get("tiles", []) for tile in row
                       if isinstance(tile, dict) and tile.get("animal") == s)
                for s in goals
            }
            animal_n = sum(occupied_by_species.values())
            wheat_avail = int(shed.get("WHEAT", 0)) + sum(int(inv.get("WHEAT", 0)) for inv in inventories)
            wheat_price = int(obs.get("market", {}).get("prices", {}).get("WHEAT", 25))
            if animal_n > 0 and wheat_avail < animal_n and money - self.config["cash_reserve"] >= wheat_price:
                wheat_buy = min(10, (animal_n * 2) - wheat_avail)
                if wheat_buy > 0:
                    market_orders.append(["BUY_PRODUCT", "WHEAT", wheat_buy])

        # E. Sales with anti-shock & town absorption
        shed = private.get("shed", {})
        prices = obs.get("market", {}).get("prices", {})
        inventories_mkt = obs.get("market", {}).get("inventory", {})
        final_turn = step >= FINAL_ACTION_STEP
        stagger_phase = step >= (FINAL_ACTION_STEP - 48)  # Last 2 days

        unlocked_shops = obs.get("town", {}).get("unlocked_shops", [])
        drain_rates = {item: 1.0 for item in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]}
        for sh in unlocked_shops:
            for it, qty in SHOP_DEMAND.get(sh, {}).items():
                drain_rates[it] = drain_rates.get(it, 0.0) + (qty * 6.0)

        animal_cnt = sum(1 for row in farm.get("tiles", []) for tile in row if isinstance(tile, dict) and tile.get("animal"))
        wheat_res = animal_cnt * 2 if not final_turn else 0

        for item, qty in shed.items():
            qty = int(qty)
            if qty <= 0:
                continue
            if item == "WHEAT" and not final_turn:
                qty = max(0, qty - wheat_res)
                if qty <= 0:
                    continue

            p = prices.get(item, 0)
            if p <= 0:
                continue

            base = CROP_INFO.get(item, {}).get("base_price") or {"EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}.get(item, p)
            min_ratio = 0.15 if stagger_phase else self.config["sell_price_ratio"]
            if not final_turn and p < base * min_ratio:
                continue

            has_town_drain = drain_rates.get(item, 1.0) > 1.5
            inv = inventories_mkt.get(item)
            if inv is None:
                inv = estimate_inventory_from_price(item, p)

            if final_turn:
                batch = qty
            elif stagger_phase:
                # In final 48 turns: sell in small steady batches to capture recovery between turns
                batch = min(qty, 4 if item in {"MILK", "WOOL"} else 6)
            elif has_town_drain:
                batch = optimal_sale_batch(item, qty, inv, min_marginal_ratio=0.55)
                batch = min(batch, 8 if item in {"MILK", "WOOL", "MELON"} else 12)
            else:
                batch = min(qty, int(self.config.get("sell_batch_size", 10)))

            if batch > 0:
                market_orders.append(["SELL", item, batch])

        # ----------------------------------------------------
        # 2. Main Farmer Action
        # ----------------------------------------------------
        tile = farm["tiles"][fy][fx]
        farmer_action = None

        # A. Standing on weed -> DIG
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            farmer_action = ["DIG"]

        # B. Pasture Building (Fix: check structure_count against sum(goals), NOT + stock)
        if farmer_action is None and goals and is_investment_phase:
            structures = {ANIMAL_INFO[s]["structure"] for s in goals}
            structure_count = sum(
                1 for row in farm.get("tiles", []) for tile in row
                if isinstance(tile, dict) and tile.get("kind") in structures
            )
            if structure_count < sum(goals.values()):
                empty = find_nearest_plant(farm, farmer_pos)
                if empty is not None:
                    build_species = next((s for s in ("COW", "SHEEP", "GOOSE") if s in goals), next(iter(goals)))
                    struct_kind = ANIMAL_INFO[build_species]["structure"]
                    farmer_action = (["BUILD_" + struct_kind] if farmer_pos == empty else next_step_toward(farmer_pos, empty))

        # C. Harvest ready crop
        if farmer_action is None:
            harvest = find_harvest_target(farm, day, farmer_pos)
            if harvest is not None:
                farmer_action = ["HARVEST"] if farmer_pos == harvest else next_step_toward(farmer_pos, harvest)

        # D. Water urgent crop
        if farmer_action is None:
            water = find_urgent_water_target(farm, farmer_pos)
            if water is not None:
                farmer_action = ["WATER"] if farmer_pos == water else next_step_toward(farmer_pos, water)

        # E. Plant crop
        if farmer_action is None:
            plant_target = find_nearest_plant(farm, farmer_pos)
            if plant_target is not None:
                curr_crop = choose_enhanced_crop(obs, self.config)
                if seeds.get(curr_crop, 0) > 0:
                    farmer_action = (["PLANT", curr_crop] if farmer_pos == plant_target else next_step_toward(farmer_pos, plant_target))

        # F. Clear nearest weed if idle
        if farmer_action is None:
            weeds = find_weeds(farm)
            if weeds:
                nearest_weed = min(weeds, key=lambda w: distance(farmer_pos, w))
                farmer_action = ["DIG"] if farmer_pos == nearest_weed else next_step_toward(farmer_pos, nearest_weed)

        if farmer_action is None:
            farmer_action = ["PASS"]

        # ----------------------------------------------------
        # 3. Farm Hands Actions
        # ----------------------------------------------------
        hand_actions: List[List[Any]] = []
        animal_acts = enhanced_animal_hand_actions(obs, farm, self.config)

        weeds = find_weeds(farm)

        for h_idx, hand in enumerate(farm.get("hands", [])):
            act = animal_acts[h_idx] if h_idx < len(animal_acts) else None
            if act is not None:
                hand_actions.append(act)
                continue

            h_pos = (hand[0], hand[1])
            h_tile = farm["tiles"][h_pos[1]][h_pos[0]]

            # Dig standing weed
            if isinstance(h_tile, dict) and h_tile.get("kind") == "WEED":
                hand_actions.append(["DIG"])
                continue

            # Water
            water = find_urgent_water_target(farm, h_pos)
            if water is not None:
                hand_actions.append(["WATER"] if h_pos == water else next_step_toward(h_pos, water))
                continue

            # Harvest
            harvest = find_harvest_target(farm, day, h_pos)
            if harvest is not None:
                hand_actions.append(["HARVEST"] if h_pos == harvest else next_step_toward(h_pos, harvest))
                continue

            # Dig weed
            if weeds:
                nearest_w = min(weeds, key=lambda w: distance(h_pos, w))
                hand_actions.append(["DIG"] if h_pos == nearest_w else next_step_toward(h_pos, nearest_w))
                continue

            hand_actions.append(["PASS"])

        # Final turn liquidation override
        if final_turn:
            liq_orders, liq_actions = submission.final_turn_liquidation(obs, farm)
            return {
                "farmer": liq_actions[0] if liq_actions else farmer_action,
                "hands": liq_actions[1:] if len(liq_actions) > 1 else hand_actions,
                "market": liq_orders[:10],
            }

        return {
            "farmer": farmer_action,
            "hands": hand_actions,
            "market": market_orders[:10],
        }


_ENHANCED_INSTANCE = EnhancedAgent()

def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return _ENHANCED_INSTANCE(obs)
