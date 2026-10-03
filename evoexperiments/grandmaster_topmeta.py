"""Grandmaster Top-Meta Agent (Exact Replication of #1 and #2 Kaggle Bots).

Key Systems:
1. ANIMAL_BLUEPRINT: Exactly 24 animals placed chronologically in columns 3-7 around the shed.
2. Pipelined Animal Delivery: Unit picks up animal from shed, walks to target slot, builds structure, and immediately places the animal (100% populated, 0 empty structures).
3. Early Fertilizer Micro-Liquidation: Sells 1 fertilizer per turn during Days 1-6 ($500-$1000 daily early cash flow).
4. 4-Quadrant Fast Expansion: NE (Day 7, $1k), SW (Day 9, $2k), SE (Day 11, $4k) funded by Day 10 Melon harvest surge.
5. Self-Sustaining Wheat Engine: 35-45 outer tiles in Wheat (feeds all 24 animals with large surplus).
6. Fertilizer Yield Multiplier: Collects 24 fertilizer daily from animals and fertilizes living crops.
7. Coordinated 12-Hand Workforce & Step 719 10k Full Liquidation.
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

from submission import (
    ANIMAL_INFO,
    CROP_INFO,
    FINAL_ACTION_STEP,
    MARKET_PARAMS,
    SHOP_DEMAND,
    calculate_price,
    count_crop,
    count_empty_tiles,
    distance,
    estimate_inventory_from_price,
    final_turn_liquidation,
    find_harvest_target,
    find_weeds,
    next_land_cost,
    next_step_toward,
    optimal_sale_batch,
    plant_tiles,
    unlocked_empty_tiles,
)

# ------------------------------------------------------------
# Exact 24-Animal Chronological Blueprint (Replicated from Replay P1)
# ------------------------------------------------------------
ANIMAL_BLUEPRINT: List[Tuple[Tuple[int, int], str, str, int]] = [
    # (pos, species, structure_kind, unlock_quadrant_count)
    # Day 0 Opening - NW Quadrant (5 animals)
    ((4, 4), "COW", "PASTURE", 1),    # Step 4 (D0H4)
    ((3, 4), "SHEEP", "PASTURE", 1),  # Step 6 (D0H6)
    ((4, 3), "SHEEP", "PASTURE", 1),  # Step 7 (D0H7)
    ((3, 3), "SHEEP", "PASTURE", 1),  # Step 8 (D0H8)
    ((4, 2), "COW", "PASTURE", 1),    # Step 10 (D0H10)

    # Days 2-4 NW Quadrant Expansion (3 cows)
    ((3, 2), "COW", "PASTURE", 1),    # Step 64 (D2H16)
    ((4, 1), "COW", "PASTURE", 1),    # Step 88 (D3H16)
    ((3, 1), "COW", "PASTURE", 1),    # Step 113 (D4H17)

    # Days 6-7 NE Quadrant Expansion (9 geese)
    ((5, 3), "GOOSE", "COOP", 2),     # Step 155 (D6H11)
    ((6, 4), "GOOSE", "COOP", 2),     # Step 156 (D6H12)
    ((5, 2), "GOOSE", "COOP", 2),     # Step 158 (D6H14)
    ((5, 4), "GOOSE", "COOP", 2),     # Step 158 (D6H14)
    ((5, 1), "GOOSE", "COOP", 2),     # Step 159 (D6H15)
    ((6, 3), "GOOSE", "COOP", 2),     # Step 163 (D6H19)
    ((7, 3), "GOOSE", "COOP", 2),     # Step 165 (D6H21)
    ((8, 4), "GOOSE", "COOP", 2),     # Step 166 (D6H22)
    ((8, 3), "GOOSE", "COOP", 2),     # Step 179 (D7H11)

    # Days 8-10 SW & SE Boundary Expansion (3 animals)
    ((3, 5), "GOOSE", "COOP", 3),     # Step 208 (D8H16)
    ((4, 5), "COW", "PASTURE", 3),     # Step 227 (D9H11)
    ((5, 5), "GOOSE", "COOP", 4),     # Step 255 (D10H15)

    # Day 12 Final 4 Sheep Expansion
    ((4, 6), "SHEEP", "PASTURE", 3),  # Step 295 (D12H7)
    ((6, 5), "SHEEP", "PASTURE", 4),  # Step 295 (D12H7)
    ((5, 6), "SHEEP", "PASTURE", 4),  # Step 297 (D12H9)
    ((2, 4), "SHEEP", "PASTURE", 1),  # Step 299 (D12H11)
]

ANIMAL_TILES_SET = {pos for pos, _, _, _ in ANIMAL_BLUEPRINT}
SHED_ADJACENT = [(4, 4), (5, 4), (4, 5), (5, 5)]


def tile_quadrant(pos: Tuple[int, int]) -> str:
    x, y = pos
    if x < 5 and y < 5:
        return "NW"
    elif x >= 5 and y < 5:
        return "NE"
    elif x < 5 and y >= 5:
        return "SW"
    else:
        return "SE"


def get_quad_count(pos: Tuple[int, int]) -> int:
    x, y = pos
    if x < 5 and y < 5:
        return 1
    elif x >= 5 and y < 5:
        return 2
    elif x < 5 and y >= 5:
        return 3
    else:
        return 4


def sort_by_quad_and_dist(candidates: List[Tuple[int, int]], assigned_quad: Optional[str], unit_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
    return sorted(
        candidates,
        key=lambda p: (0 if assigned_quad and tile_quadrant(p) == assigned_quad else 1, distance(unit_pos, p))
    )


class GrandmasterTopMetaAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def get_max_hands(self, day: int, quads: int = 1) -> int:
        if day == 0:
            return 4
        elif day == 1:
            return 4
        elif day == 2:
            return 5
        elif day in (3, 4, 5):
            return 6
        elif quads == 2:
            return 8
        elif quads == 3:
            return 10
        elif quads >= 4:
            return 13
        return 6

    def plan_market_orders(self, obs: Dict[str, Any], farm: Dict[str, Any]) -> List[List[Any]]:
        market_orders: List[List[Any]] = []
        step = int(obs.get("step", 0))
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        money = int(farm.get("money", 0))
        private = obs.get("private", {})
        shed = dict(private.get("shed", {}))
        seeds = dict(private.get("seeds", {}))
        prices = obs.get("market", {}).get("prices", {})
        inventories = [dict(inv) for inv in private.get("inventories", [])]
        unlocked_quads = farm.get("unlocked_quadrants", ["NW"])
        quads = len(unlocked_quads)

        # 1. Day 0 Exact Master Opening (Replicated from Replay P1)
        if day == 0:
            if step == 1 and money >= 2900:
                return [["BUY_ANIMAL", "COW", 1], ["BUY_PRODUCT", "WHEAT", 5]]
            elif step == 2 and money >= 2000:
                return [
                    ["SELL", "WHEAT", 1],
                    ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"],
                    ["BUY_ANIMAL", "COW", 1],
                    ["BUY_ANIMAL", "SHEEP", 3],
                ]
            elif step in (4, 5, 6, 7) and shed.get("WHEAT", 0) < 4:
                return [["BUY_PRODUCT", "WHEAT", 1]]
            elif step in (7, 8, 12) and seeds.get("MELON", 0) + count_crop(farm, "MELON") < 12 and money >= 160:
                return [["BUY_SEED", "MELON", 2]]
            elif step >= 11 and step <= 23 and money >= 20:
                wheat_total = seeds.get("WHEAT", 0) + count_crop(farm, "WHEAT")
                if wheat_total < 18:
                    return [["BUY_SEED", "WHEAT", 1]]
            return []

        # Count live animals on farm
        cows_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "COW")
        sheep_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "SHEEP")
        geese_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "GOOSE")
        total_anim = cows_now + sheep_now + geese_now
        unfed_count = sum(
            1 for ax, ay in ANIMAL_TILES_SET
            if 0 <= ay < 10 and 0 <= ax < 10 and isinstance(farm.get("tiles", [])[ay][ax], dict)
            and farm.get("tiles", [])[ay][ax].get("animal") and not farm.get("tiles", [])[ay][ax].get("fed_today", False)
        )

        # 2. Daily Hiring Burst at Hour 0 and Hour 1
        hires_today = int(farm.get("hires_today", 0))
        target_hands = self.get_max_hands(day, quads)
        FIB_COSTS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
        
        if hour in (0, 1) and hires_today < target_hands:
            cur_money = money
            hires_to_add = 0
            while hires_today + hires_to_add < target_hands:
                next_hire_cost = FIB_COSTS[hires_today + hires_to_add] if hires_today + hires_to_add < len(FIB_COSTS) else 233
                if cur_money >= next_hire_cost and hires_to_add < 8:
                    cur_money -= next_hire_cost
                    hires_to_add += 1
                else:
                    break
            for _ in range(hires_to_add):
                market_orders.append(["HIRE"])

        # 3. Continuous Early Fertilizer Micro-Selling (Days 1-8)
        fert_qty = shed.get("FERTILIZER", 0)
        fert_to_sell = 0
        if fert_qty > 0 and day < 9 and len(market_orders) < 9:
            fert_to_sell = min(2, fert_qty)
            market_orders.append(["SELL", "FERTILIZER", fert_to_sell])

        # 4. Feed Protection: Buy wheat product immediately if reserves dip!
        wheat_in_shed = shed.get("WHEAT", 0)
        wheat_carried = sum(inv.get("WHEAT", 0) for inv in inventories)
        wheat_avail = wheat_in_shed + wheat_carried
        effective_money = money + fert_to_sell * 45

        # Critical: On Days 1-10, ensure wheat buffer so handlers can feed on arrival
        if day in (1, 2) and wheat_avail < 8 and effective_money >= 18 and len(market_orders) < 9:
            needed = min(4, 8 - wheat_avail)
            if needed > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", needed])
        elif day <= 10 and total_anim > 0 and wheat_avail < (total_anim + 4) and effective_money >= 18 and len(market_orders) < 9:
            needed = min(8, (total_anim + 4) - wheat_avail)
            if needed > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", needed])
        elif day > 10 and unfed_count > 0 and wheat_avail < unfed_count and money >= 25 and len(market_orders) < 9:
            # True emergency on Day 11+: Not enough wheat on farm to feed today's hungry animals!
            needed = unfed_count - wheat_avail
            market_orders.append(["BUY_PRODUCT", "WHEAT", min(8, needed)])

        # Day 1: Purchase remaining melons to hit 12 melons
        if day == 1 and money >= 160 and seeds.get("MELON", 0) + count_crop(farm, "MELON") < 12 and len(market_orders) < 9:
            market_orders.append(["BUY_SEED", "MELON", 2])

        # 5. Day 6 Wool Liquidation to trigger NE Land Expansion!
        wool_in_shed = shed.get("WOOL", 0)
        if day >= 6 and wool_in_shed > 0 and quads == 1 and len(market_orders) < 8:
            market_orders.append(["SELL", "WOOL", wool_in_shed])

        # 6. Dynamic Product Selling (Days 6+)
        if day >= 6:
            for item in ["MELON", "WOOL", "MILK", "EGG", "STRAWBERRY", "CARROT", "TOMATO"]:
                qty = shed.get(item, 0)
                if qty <= 0:
                    continue
                if len(market_orders) < 9:
                    market_orders.append(["SELL", item, min(16, qty)])

            # Fertilizer selling on Day 10+ (maintain 10 reserve for active crop fertilizing)
            fert_in_shed = shed.get("FERTILIZER", 0)
            if day >= 10 and fert_in_shed > 10 and len(market_orders) < 9:
                market_orders.append(["SELL", "FERTILIZER", min(10, fert_in_shed - 10)])

        # Wheat selling: Sell excess wheat at any hour when above reserve, or full liquidation on Day 29!
        if day == 29:
            # End of season liquidation!
            for item in ["MELON", "WOOL", "MILK", "EGG", "STRAWBERRY", "CARROT", "TOMATO", "FERTILIZER", "WHEAT"]:
                qty = shed.get(item, 0)
                if qty > 0 and len(market_orders) < 9:
                    market_orders.append(["SELL", item, min(24, qty)])
        else:
            feed_reserve = total_anim + 4
            if wheat_in_shed > feed_reserve and len(market_orders) < 9:
                market_orders.append(["SELL", "WHEAT", min(16, wheat_in_shed - feed_reserve)])

        # 7. 4-Quadrant Fast Expansion (NE Day 6, SW Day 8, SE Day 10)
        if quads < 4 and len(market_orders) < 9:
            cost = next_land_cost(farm)
            if cost is not None:
                wool_est = shed.get("WOOL", 0) * 150 if day >= 6 else 0
                milk_est = shed.get("MILK", 0) * 150 if day >= 8 else 0
                melon_est = shed.get("MELON", 0) * 100 if day >= 10 else 0
                if quads == 1 and day >= 6 and (money + wool_est) >= cost:
                    market_orders.append(["BUY_LAND"])
                elif quads == 2 and day >= 8 and (money + milk_est) >= cost:
                    market_orders.append(["BUY_LAND"])
                elif quads == 3 and day >= 10 and (money + melon_est) >= cost:
                    market_orders.append(["BUY_LAND"])

        # 8. Animal Purchases for the 24-Animal Blueprint
        if day <= 18:
            tiles = farm.get("tiles", [])
            placed_positions = {
                (x, y) for y, r in enumerate(tiles) for x, t in enumerate(r)
                if isinstance(t, dict) and t.get("animal")
            }
            in_transit = sum(1 for s in ["COW", "SHEEP", "GOOSE"] for inv in inventories if inv.get(s, 0) > 0) + \
                         sum(shed.get(s, 0) for s in ["COW", "SHEEP", "GOOSE"])

            for pos, species, struct_kind, req_quads in ANIMAL_BLUEPRINT:
                if quads >= req_quads and pos not in placed_positions:
                    buffer = 20 if day < 5 else 60
                    cost = ANIMAL_INFO[species]["cost"] + buffer
                    if in_transit < 2 and money >= cost and len(market_orders) < 9:
                        market_orders.append(["BUY_ANIMAL", species, 1])
                        in_transit += 1
                    break

        # 9. Dynamic Seed Replanting Pipeline (replacing harvested one-time crops)
        if day < 28:
            active_wheat = count_crop(farm, "WHEAT")
            active_straw = count_crop(farm, "STRAWBERRY")
            active_carrot = count_crop(farm, "CARROT")
            active_tomato = count_crop(farm, "TOMATO")

            seed_wheat = seeds.get("WHEAT", 0)
            seed_straw = seeds.get("STRAWBERRY", 0)
            seed_carrot = seeds.get("CARROT", 0)
            seed_tomato = seeds.get("TOMATO", 0)

            # Strawberry & Tomato (ongoing crops, plant once, produce forever)
            if quads >= 2 and (active_straw + seed_straw) < 24 and money >= 80 and len(market_orders) < 9:
                market_orders.append(["BUY_SEED", "STRAWBERRY", min(4, 24 - (active_straw + seed_straw))])

            if quads >= 3 and (active_tomato + seed_tomato) < 8 and money >= 40 and len(market_orders) < 9:
                market_orders.append(["BUY_SEED", "TOMATO", min(2, 8 - (active_tomato + seed_tomato))])

            # Wheat & Carrot (one-time crops: continuously replant harvested tiles!)
            target_active_wheat = 16 if quads == 1 else (22 if quads <= 3 else 30)
            if active_wheat < target_active_wheat and seed_wheat < 8 and money >= 30 and len(market_orders) < 9:
                needed = min(8, (target_active_wheat - active_wheat) + 4 - seed_wheat)
                if needed > 0:
                    market_orders.append(["BUY_SEED", "WHEAT", needed])

            if quads >= 3 and day < 26:
                target_active_carrot = 15 if quads == 3 else 25
                if active_carrot < target_active_carrot and seed_carrot < 8 and money >= 50 and len(market_orders) < 9:
                    needed = min(6, (target_active_carrot - active_carrot) + 4 - seed_carrot)
                    if needed > 0:
                        market_orders.append(["BUY_SEED", "CARROT", needed])

        return market_orders[:10]

    def plan_units(self, obs: Dict[str, Any], farm: Dict[str, Any]) -> Tuple[List[Any], List[List[Any]]]:
        farmer_pos = tuple(farm["farmer"])
        hands = farm.get("hands", []) or []
        all_units = [farmer_pos] + [tuple(h) for h in hands]
        inventories = [dict(inv) for inv in obs.get("private", {}).get("inventories", [{} for _ in all_units])]
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        step = int(obs.get("step", 0))
        shed = dict(obs.get("private", {}).get("shed", {}))
        seeds = dict(obs.get("private", {}).get("seeds", {}))
        unlocked_quads = farm.get("unlocked_quadrants", ["NW"])
        quads = len(unlocked_quads)
        tiles = farm.get("tiles", [])

        actions: List[List[Any]] = [["PASS"] for _ in all_units]
        claimed_targets = set()

        # Build animal status lookup
        placed_positions = {
            (x, y): t.get("animal") for y, r in enumerate(tiles) for x, t in enumerate(r)
            if isinstance(t, dict) and t.get("animal")
        }

        # Determine animal handler count: dynamically scale with animal population!
        total_anim_count = len(placed_positions)
        if total_anim_count <= 5:
            handler_cutoff = 2
        elif total_anim_count <= 9:
            handler_cutoff = 3
        elif total_anim_count <= 16:
            handler_cutoff = 5
        else:
            handler_cutoff = 6

        # Count unfed animals on farm
        unfed_count = sum(
            1 for ax, ay in ANIMAL_TILES_SET
            if 0 <= ay < 10 and 0 <= ax < 10 and isinstance(tiles[ay][ax], dict)
            and tiles[ay][ax].get("animal") and not tiles[ay][ax].get("fed_today", False)
        )

        # ----------------------------------------------------
        # Pass 1: IMMEDIATE CURRENT TILE ACTIONS
        # ----------------------------------------------------
        for u_idx, pos in enumerate(all_units):
            inv = inventories[u_idx] if u_idx < len(inventories) else {}
            tile = tiles[pos[1]][pos[0]] if 0 <= pos[1] < 10 and 0 <= pos[0] < 10 else None
            is_handler = (u_idx < handler_cutoff)

            # 1. At Shed Actions
            if pos in SHED_ADJACENT:
                carried_wool_milk = inv.get("WOOL", 0) + inv.get("MILK", 0)
                carried_harvest = sum(inv.get(prod, 0) for prod in ["WOOL", "MILK", "EGG", "MELON", "STRAWBERRY", "CARROT", "TOMATO"])
                # Immediate drop off for wool/milk or large egg amounts to fund expansions!
                if carried_wool_milk > 0 or inv.get("EGG", 0) >= 4:
                    actions[u_idx] = ["DROP"]
                    continue
                elif is_handler and unfed_count > 0:
                    pass
                else:
                    # In days 1-9, drop fertilizer immediately to fund wheat buys!
                    if day < 10 and inv.get("FERTILIZER", 0) > 0:
                        actions[u_idx] = ["DROP"]
                        continue
                    # If carrying harvest produce, or carrying wheat when all animals are fed: DROP!
                    if carried_harvest > 0 or (inv.get("WHEAT", 0) > 0 and unfed_count == 0):
                        actions[u_idx] = ["DROP"]
                        continue

                # Pickup wheat feed: ONLY ANIMAL HANDLERS pick up wheat feed when animals need food!
                if is_handler and unfed_count > 0 and inv.get("WHEAT", 0) <= 1 and shed.get("WHEAT", 0) > 0:
                    pick_qty = min(6, shed.get("WHEAT", 0))
                    actions[u_idx] = ["PICKUP", "WHEAT", pick_qty]
                    shed["WHEAT"] -= pick_qty
                    inv["WHEAT"] = inv.get("WHEAT", 0) + pick_qty
                    continue

                # Animal Delivery Pickup: If carrying no animal, pick up the next blueprint animal from shed!
                # STRICT GUARD: If unfed_count > 0, handlers MUST feed first!
                if (not is_handler or unfed_count == 0) and inv.get("COW", 0) == 0 and inv.get("SHEEP", 0) == 0 and inv.get("GOOSE", 0) == 0:
                    for b_pos, b_spec, b_struct, b_req in ANIMAL_BLUEPRINT:
                        if quads >= b_req and b_pos not in placed_positions and b_pos not in claimed_targets:
                            if shed.get(b_spec, 0) > 0:
                                actions[u_idx] = ["PICKUP", b_spec, 1]
                                shed[b_spec] -= 1
                                inv[b_spec] = inv.get(b_spec, 0) + 1
                                claimed_targets.add(b_pos)
                                break
                    if actions[u_idx] != ["PASS"]:
                        continue

                # On Day 10+, pickup fertilizer from shed if available to fertilize outer crops!
                if not is_handler and day >= 10 and inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) >= 4:
                    take_fert = min(3, shed.get("FERTILIZER", 0))
                    actions[u_idx] = ["PICKUP", "FERTILIZER", take_fert]
                    shed["FERTILIZER"] -= take_fert
                    inv["FERTILIZER"] = inv.get("FERTILIZER", 0) + take_fert
                    continue

            # 2. Pipelined Animal Construction & Placement on Destination Tile
            for b_pos, b_spec, b_struct, b_req in ANIMAL_BLUEPRINT:
                if pos == b_pos and b_pos not in placed_positions:
                    if inv.get(b_spec, 0) > 0:
                        if tile is None:
                            actions[u_idx] = ["BUILD_PASTURE"] if b_struct == "PASTURE" else ["BUILD_COOP"]
                            continue
                        elif isinstance(tile, dict) and not tile.get("animal"):
                            actions[u_idx] = ["PLACE", b_spec, 1]
                            inv[b_spec] -= 1
                            placed_positions[b_pos] = b_spec
                            continue

            # 3. Animal Maintenance on Current Tile (ONLY ANIMAL HANDLERS)
            if is_handler and isinstance(tile, dict) and tile.get("animal"):
                # Feed animal (CRITICAL)
                if not tile.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                    actions[u_idx] = ["FEED"]
                    inv["WHEAT"] -= 1
                    continue

                # If there are still hungry animals on the farm and this unit carries wheat, DO NOT LINGER!
                # Move immediately to feed the next animal!
                if unfed_count > 0 and inv.get("WHEAT", 0) > 0:
                    pass
                else:
                    # Harvest animal product (milk, wool, eggs)
                    if tile.get("yield_units", 0) > 0:
                        actions[u_idx] = ["HARVEST"]
                        continue

                    # Collect fertilizer
                    if tile.get("fertilizer_available", False):
                        actions[u_idx] = ["COLLECT_FERTILIZER"]
                        inv["FERTILIZER"] = inv.get("FERTILIZER", 0) + 1
                        continue

                    # Care for animal (+1 care bonus)
                    if not tile.get("cared_today", False):
                        actions[u_idx] = ["CARE"]
                        continue

            # 4. Plant Maintenance on Current Tile
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile.get("crop")
                age = day - tile.get("planted_day", 0)
                first_yield = CROP_INFO[crop]["first_yield_day"]
                yield_units = tile.get("yield_units", 0)

                # Harvest mature crop
                if age >= first_yield and yield_units > 0:
                    actions[u_idx] = ["HARVEST"]
                    continue

                # Water unwatered crop
                if not tile.get("watered_today", False):
                    actions[u_idx] = ["WATER"]
                    continue

                # Fertilize living crop if unit carries fertilizer (ONLY on Day 10+)
                fertilized_until = tile.get("fertilized_until_day", -1)
                if day >= 10 and inv.get("FERTILIZER", 0) > 0 and fertilized_until <= day and day < 27:
                    actions[u_idx] = ["FERTILIZE"]
                    inv["FERTILIZER"] -= 1
                    continue

            # 5. Plant Seeds on Current Empty Tile
            if tile is None and pos not in ANIMAL_TILES_SET:
                x, y = pos
                zone_crop = "WHEAT"
                if y <= 1:
                    zone_crop = "STRAWBERRY"
                elif y >= 7 and x <= 2:
                    zone_crop = "TOMATO"
                elif y >= 7 and x >= 3:
                    zone_crop = "CARROT"

                # Check melon override on Days 0-2
                if day <= 2 and seeds.get("MELON", 0) > 0 and (x in (1, 2) or y in (1, 2)):
                    zone_crop = "MELON"

                actual_crop = zone_crop if seeds.get(zone_crop, 0) > 0 else None
                if actual_crop is None:
                    for cand in ["WHEAT", "CARROT", "STRAWBERRY", "TOMATO"]:
                        if seeds.get(cand, 0) > 0:
                            actual_crop = cand
                            break

                if actual_crop and seeds.get(actual_crop, 0) > 0:
                    actions[u_idx] = ["PLANT", actual_crop]
                    seeds[actual_crop] -= 1
                    claimed_targets.add(pos)
                    continue

            # 6. Dig Weed on Current Tile
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                actions[u_idx] = ["DIG"]
                continue

        # ----------------------------------------------------
        # Pass 2: ROUTING UNASSIGNED UNITS TO TARGETS
        # ----------------------------------------------------
        unassigned = [i for i, a in enumerate(actions) if a == ["PASS"]]

        for u_idx in unassigned:
            pos = all_units[u_idx]
            inv = inventories[u_idx] if u_idx < len(inventories) else {}
            is_handler = (u_idx < handler_cutoff)
            field_idx = max(0, u_idx - handler_cutoff)
            assigned_quad = unlocked_quads[field_idx % len(unlocked_quads)] if not is_handler else None

            # 1. Animal Delivery In-Transit: If carrying an animal, walk straight to its blueprint slot!
            carried_animal = None
            for a_name in ["COW", "SHEEP", "GOOSE"]:
                if inv.get(a_name, 0) > 0:
                    carried_animal = a_name
                    break

            if carried_animal and not (inv.get("WHEAT", 0) > 0 and unfed_count > 0):
                for b_pos, b_spec, b_struct, b_req in ANIMAL_BLUEPRINT:
                    if b_spec == carried_animal and b_pos not in placed_positions:
                        if distance(pos, b_pos) > 0:
                            claimed_targets.add(b_pos)
                            actions[u_idx] = next_step_toward(pos, b_pos)
                            break
                if actions[u_idx] != ["PASS"]:
                    continue

            # 2. Drop off high-value produce / fertilizer to shed
            carried_wool_milk = inv.get("WOOL", 0) + inv.get("MILK", 0)
            if carried_wool_milk > 0:
                target_shed = min(SHED_ADJACENT, key=lambda s: distance(pos, s))
                if distance(pos, target_shed) > 0:
                    actions[u_idx] = next_step_toward(pos, target_shed)
                    continue
            elif is_handler and unfed_count > 0 and inv.get("WHEAT", 0) > 0 and sum(inv.values()) < 8:
                pass  # Keep feeding! Do not return to shed early!
            else:
                high_val_produce = sum(inv.get(p, 0) for p in ["WOOL", "MILK", "EGG", "MELON"])
                carried_fert = inv.get("FERTILIZER", 0)
                carried_wheat = inv.get("WHEAT", 0) if (not is_handler or unfed_count == 0) else 0
                carried_crops = sum(inv.get(p, 0) for p in ["STRAWBERRY", "CARROT", "TOMATO"]) + carried_wheat

                should_drop = (
                    high_val_produce > 0
                    or (day < 10 and carried_fert > 0)
                    or carried_crops >= 6
                )
                if should_drop:
                    target_shed = min(SHED_ADJACENT, key=lambda s: distance(pos, s))
                    if distance(pos, target_shed) > 0:
                        actions[u_idx] = next_step_toward(pos, target_shed)
                        continue

            # 3. ANIMAL CHORES (ONLY ANIMAL HANDLERS)
            if is_handler:
                # 3a. Animal Feeding (CRITICAL FIRST PRIORITY): Route units carrying wheat to unfed animals
                if inv.get("WHEAT", 0) > 0:
                    unfed = []
                    for ax, ay in ANIMAL_TILES_SET:
                        if (ax, ay) in claimed_targets:
                            continue
                        if 0 <= ay < 10 and 0 <= ax < 10 and isinstance(tiles[ay][ax], dict):
                            at = tiles[ay][ax]
                            if at.get("animal") and not at.get("fed_today", False):
                                is_emergency = 0 if at.get("consecutive_unfed", 0) >= 1 else 1
                                unfed.append((is_emergency, distance(pos, (ax, ay)), (ax, ay)))
                    if unfed:
                        unfed.sort()
                        _, d_u, u_pos = unfed[0]
                        if d_u > 0:
                            claimed_targets.add(u_pos)
                            actions[u_idx] = next_step_toward(pos, u_pos)
                            continue

                # 3b. Route to shed to pick up wheat feed if animals are still unfed
                has_unfed = any(
                    isinstance(tiles[ay][ax], dict) and tiles[ay][ax].get("animal") and not tiles[ay][ax].get("fed_today", False)
                    for ax, ay in ANIMAL_TILES_SET if 0 <= ay < 10 and 0 <= ax < 10
                )
                if inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0 and has_unfed:
                    target_shed = min(SHED_ADJACENT, key=lambda s: distance(pos, s))
                    if distance(pos, target_shed) > 0:
                        actions[u_idx] = next_step_toward(pos, target_shed)
                        continue

                # 3c. Animal Product Harvesting (WOOL, MILK, EGG)
                anim_harvests = []
                for ax, ay in ANIMAL_TILES_SET:
                    if (ax, ay) in claimed_targets:
                        continue
                    if 0 <= ay < 10 and 0 <= ax < 10 and isinstance(tiles[ay][ax], dict):
                        at = tiles[ay][ax]
                        if at.get("animal") and at.get("yield_units", 0) > 0:
                            anim_harvests.append((distance(pos, (ax, ay)), (ax, ay)))
                if anim_harvests:
                    anim_harvests.sort()
                    d_ah, ah_pos = anim_harvests[0]
                    if d_ah > 0:
                        claimed_targets.add(ah_pos)
                        actions[u_idx] = next_step_toward(pos, ah_pos)
                        continue

                # 3d. Animal Care
                uncared = []
                for ax, ay in ANIMAL_TILES_SET:
                    if (ax, ay) in claimed_targets:
                        continue
                    if 0 <= ay < 10 and 0 <= ax < 10 and isinstance(tiles[ay][ax], dict):
                        at = tiles[ay][ax]
                        if at.get("animal") and not at.get("cared_today", False):
                            uncared.append((distance(pos, (ax, ay)), (ax, ay)))
                if uncared:
                    uncared.sort()
                    d_c, c_pos = uncared[0]
                    if d_c > 0:
                        claimed_targets.add(c_pos)
                        actions[u_idx] = next_step_toward(pos, c_pos)
                        continue

                # 3e. Animal Fertilizer Collection
                fert_anims = []
                for ax, ay in ANIMAL_TILES_SET:
                    if (ax, ay) in claimed_targets:
                        continue
                    if 0 <= ay < 10 and 0 <= ax < 10 and isinstance(tiles[ay][ax], dict):
                        at = tiles[ay][ax]
                        if at.get("animal") and at.get("fertilizer_available", False):
                            fert_anims.append((distance(pos, (ax, ay)), (ax, ay)))
                if fert_anims:
                    fert_anims.sort()
                    d_f, f_pos = fert_anims[0]
                    if d_f > 0:
                        claimed_targets.add(f_pos)
                        actions[u_idx] = next_step_toward(pos, f_pos)
                        continue

            # 4. FIELD & CROP TASKS (For Field Farmers, and Handlers when animal chores complete)

            # 4a. Melon Harvest Rush (Days 10-12 ONLY when age >= 10 and yield > 0)
            if day >= 10:
                mature_melons = [
                    (distance(pos, (x, y)), (x, y))
                    for x, y, crop in plant_tiles(farm, "MELON")
                    if (x, y) not in claimed_targets
                    and farm["tiles"][y][x].get("yield_units", 0) > 0
                    and (day - farm["tiles"][y][x].get("planted_day", 0)) >= 10
                ]
                if mature_melons:
                    mature_melons.sort()
                    d_m, m_pos = mature_melons[0]
                    if d_m > 0:
                        claimed_targets.add(m_pos)
                        actions[u_idx] = next_step_toward(pos, m_pos)
                        continue

            # 4b. Emergency Plant Watering (consecutive_unwatered >= 1)
            emergency_waters = []
            for px, py, pcrop in plant_tiles(farm):
                if (px, py) in claimed_targets:
                    continue
                ptile = tiles[py][px]
                if not ptile.get("watered_today", False) and ptile.get("consecutive_unwatered", 0) >= 1:
                    emergency_waters.append((px, py))
            if emergency_waters:
                sorted_ew = sort_by_quad_and_dist(emergency_waters, assigned_quad, pos)
                w_pos = sorted_ew[0]
                if distance(pos, w_pos) > 0:
                    claimed_targets.add(w_pos)
                    actions[u_idx] = next_step_toward(pos, w_pos)
                    continue

            # 4c. Harvest ready crops (frees up tiles and brings produce into inventory!)
            harvest_ready = []
            for px, py, pcrop in plant_tiles(farm):
                if (px, py) in claimed_targets:
                    continue
                ptile = tiles[py][px]
                age = day - ptile.get("planted_day", day)
                first_yield = CROP_INFO.get(pcrop, {}).get("first_yield_day", 999)
                if ptile.get("yield_units", 0) > 0 and age >= first_yield:
                    harvest_ready.append((px, py))
            if harvest_ready:
                sorted_h = sort_by_quad_and_dist(harvest_ready, assigned_quad, pos)
                h_pos = sorted_h[0]
                if distance(pos, h_pos) > 0:
                    claimed_targets.add(h_pos)
                    actions[u_idx] = next_step_toward(pos, h_pos)
                    continue

            # 4d. Plant available seeds on empty tiles
            def get_target_empty_tile():
                pref = []
                fall = []
                for ex, ey in unlocked_empty_tiles(farm):
                    if (ex, ey) in ANIMAL_TILES_SET or (ex, ey) in claimed_targets:
                        continue
                    fall.append((ex, ey))
                    if ey <= 1 and seeds.get("STRAWBERRY", 0) > 0:
                        pref.append((ex, ey))
                    elif ey >= 7 and ex <= 2 and seeds.get("TOMATO", 0) > 0:
                        pref.append((ex, ey))
                    elif ey >= 7 and seeds.get("CARROT", 0) > 0:
                        pref.append((ex, ey))
                    elif ey not in (0, 1, 7, 8, 9) and seeds.get("WHEAT", 0) > 0:
                        pref.append((ex, ey))
                pool = pref if pref else fall
                if pool:
                    sorted_pool = sort_by_quad_and_dist(pool, assigned_quad, pos)
                    return sorted_pool[0]
                return None

            if sum(seeds.values()) > 0:
                target_empty = get_target_empty_tile()
                if target_empty is not None and distance(pos, target_empty) > 0:
                    claimed_targets.add(target_empty)
                    actions[u_idx] = next_step_toward(pos, target_empty)
                    continue

            # 4e. Normal Crop Watering
            normal_waters = []
            for px, py, pcrop in plant_tiles(farm):
                if (px, py) in claimed_targets:
                    continue
                ptile = tiles[py][px]
                if not ptile.get("watered_today", False):
                    normal_waters.append((px, py))
            if normal_waters:
                sorted_nw = sort_by_quad_and_dist(normal_waters, assigned_quad, pos)
                nw_pos = sorted_nw[0]
                if distance(pos, nw_pos) > 0:
                    claimed_targets.add(nw_pos)
                    actions[u_idx] = next_step_toward(pos, nw_pos)
                    continue

            # 4f. Weed clearing (Clear weeds so tiles can be planted!)
            weeds = [w for w in find_weeds(farm) if w not in claimed_targets]
            if weeds:
                sorted_w = sort_by_quad_and_dist(weeds, assigned_quad, pos)
                near_w = sorted_w[0]
                if distance(pos, near_w) > 0:
                    claimed_targets.add(near_w)
                    actions[u_idx] = next_step_toward(pos, near_w)
                    continue

            # 4g. Active crop fertilization (ONLY on Day 10+)
            if day >= 10 and inv.get("FERTILIZER", 0) > 0 and day < 27:
                unfert_crops = []
                for px, py, pcrop in plant_tiles(farm):
                    if (px, py) in claimed_targets:
                        continue
                    ptile = tiles[py][px]
                    if ptile.get("fertilized_until_day", -1) <= day:
                        unfert_crops.append((px, py))
                if unfert_crops:
                    sorted_fc = sort_by_quad_and_dist(unfert_crops, assigned_quad, pos)
                    fc_pos = sorted_fc[0]
                    if distance(pos, fc_pos) > 0:
                        claimed_targets.add(fc_pos)
                        actions[u_idx] = next_step_toward(pos, fc_pos)
                        continue

            # 4h. Walk toward shed if idle
            target_shed = min(SHED_ADJACENT, key=lambda s: distance(pos, s))
            if distance(pos, target_shed) > 0:
                actions[u_idx] = next_step_toward(pos, target_shed)
            else:
                actions[u_idx] = ["PASS"]

        return actions[0], actions[1:]

    def __call__(self, obs: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        player = obs["player"]
        farm = obs["farms"][player]
        step = int(obs.get("step", 0))

        # Final turn 719: Send 10k liquidation
        if step >= FINAL_ACTION_STEP:
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


_agent = GrandmasterTopMetaAgent()

def agent(obs: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    return _agent(obs, *args, **kwargs)
