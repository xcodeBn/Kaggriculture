"""Top-Meta Re-engineered Agent for Kaggriculture.

Reverse-engineered from the #1 and #2 Kaggle leaderboard showdown (116330962.json, $85k+):
1. 24-Animal Central Ring:
   - 13 Pastures (Columns 3-4, NW/SW) for 6-7 Cows & 6-7 Sheep
   - 11 Coops (Columns 5-7, NE) for 11 Geese
   - Direct shed connectivity: 1-step walking distance for feed, care, fertilizer collection, and drops.
2. Continuous Fertilizer Micro-Selling:
   - Sells 1-2 fertilizer each turn during Days 1-8 to generate $500-$1000 daily early cash flow.
3. 4-Quadrant Fast Expansion:
   - Day 7: NE ($1k)
   - Day 9: SW ($2k)
   - Day 11: SE ($4k) funded by the Day 10 Melon harvest surge.
4. Self-Feeding Wheat Flanks:
   - 35-45 outer tiles dedicated to Wheat to supply the 24 daily wheat feed for free.
5. Active Crop Fertilization:
   - Collects 24 fertilizer daily from animals.
   - Actively fertilizes living wheat, strawberries, melons, and carrots to double yields.
6. Coordinated 12-Hand Workforce & Turn 719 10k Full Liquidation.
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
# Top-Meta Structural Layout
# ------------------------------------------------------------
PASTURE_SPECS = [
    # NW Quadrant (Starting)
    (4, 4), (3, 4), (4, 3), (3, 3), (4, 2), (3, 2), (4, 1), (3, 1), (4, 0),
    # SW Quadrant (Day 9+)
    (4, 5), (3, 5), (2, 5), (3, 6),
    # SE Quadrant (Day 11+)
    (5, 6), (6, 5),
]

COOP_SPECS = [
    # NE Quadrant (Day 7+)
    (5, 4), (5, 3), (6, 4), (7, 4), (5, 2), (6, 2), (7, 3), (5, 1),
    # SE Quadrant (Day 11+)
    (5, 5),
]

ALL_ANIMAL_TILES = set(PASTURE_SPECS + COOP_SPECS)
SHED_ADJACENT = [(4, 4), (5, 4), (4, 5), (5, 5)]

TOPMETA_CONFIG: Dict[str, Any] = {
    # Animals
    "cow_target_early": 2,
    "sheep_target_early": 3,
    "cow_target_mid": 6,
    "sheep_target_mid": 6,
    "goose_target_mid": 11,
    
    # Workforce
    "hands_d0": 4,
    "hands_d1": 4,
    "hands_d2": 5,
    "hands_d5": 6,
    "hands_d7": 8,
    "hands_d10": 12,

    # Land
    "expand_day_ne": 7,
    "expand_day_sw": 9,
    "expand_day_se": 11,

    # Trading
    "early_fert_batch": 1,
    "mid_sell_batch": 8,
}


def get_quadrant(pos: Tuple[int, int]) -> str:
    x, y = pos
    if x < 5 and y < 5:
        return "NW"
    elif x >= 5 and y < 5:
        return "NE"
    elif x < 5 and y >= 5:
        return "SW"
    else:
        return "SE"


def is_tile_unlocked(pos: Tuple[int, int], unlocked_quads: List[str]) -> bool:
    return get_quadrant(pos) in unlocked_quads


class TopMetaPolicy:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = dict(TOPMETA_CONFIG)
        if config:
            self.config.update(config)

    def get_max_hands(self, day: int) -> int:
        if day < 2:
            return int(self.config.get("hands_d0", 4))
        elif day < 5:
            return int(self.config.get("hands_d2", 5))
        elif day < 7:
            return int(self.config.get("hands_d5", 6))
        elif day < 10:
            return int(self.config.get("hands_d7", 8))
        else:
            return int(self.config.get("hands_d10", 12))

    def plan_market_orders(self, obs: Dict[str, Any], farm: Dict[str, Any]) -> List[List[Any]]:
        market_orders: List[List[Any]] = []
        step = int(obs.get("step", 0))
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        money = int(farm.get("money", 0))
        private = obs.get("private", {})
        shed = private.get("shed", {})
        seeds = private.get("seeds", {})
        prices = obs.get("market", {}).get("prices", {})
        inventories = private.get("inventories", [])
        m_inv = obs.get("market", {}).get("inventory", {})
        unlocked_quads = farm.get("unlocked_quadrants", ["NW"])
        quads = len(unlocked_quads)

        # ----------------------------------------------------
        # 1. Day 0 Exact Master Opening
        # ----------------------------------------------------
        if day == 0:
            if step == 1 and money >= 2900:
                return [["BUY_ANIMAL", "COW", 1], ["BUY_PRODUCT", "WHEAT", 5]]
            elif step == 2 and money >= 2500:
                return [
                    ["SELL", "WHEAT", 1],
                    ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"],
                    ["BUY_ANIMAL", "COW", 1],
                    ["BUY_ANIMAL", "SHEEP", 3],
                ]
            elif step in (7, 8, 10) and seeds.get("MELON", 0) + count_crop(farm, "MELON") < 8 and money >= 240:
                return [["BUY_SEED", "MELON", 2]]
            elif step >= 11 and step <= 21 and money >= 20:
                wheat_total = seeds.get("WHEAT", 0) + count_crop(farm, "WHEAT")
                if wheat_total < 14:
                    return [["BUY_SEED", "WHEAT", 1]]
            return []

        # ----------------------------------------------------
        # 2. Continuous Early Fertilizer Micro-Selling (Days 1-8)
        # ----------------------------------------------------
        fert_qty = shed.get("FERTILIZER", 0)
        if fert_qty > 0 and day < 9:
            # Sell 1-2 at a time to prevent any price shock
            batch = min(fert_qty, int(self.config.get("early_fert_batch", 1)))
            market_orders.append(["SELL", "FERTILIZER", batch])

        # ----------------------------------------------------
        # 3. Dynamic Product Selling (Days 6+)
        # ----------------------------------------------------
        if day >= 6:
            unlocked_shops = obs.get("town", {}).get("unlocked_shops", [])
            for item in ["MELON", "WOOL", "MILK", "EGG", "STRAWBERRY", "CARROT", "TOMATO", "FERTILIZER"]:
                qty = shed.get(item, 0)
                if qty <= 0:
                    continue
                p = prices.get(item, 0)
                base_p = MARKET_PARAMS[item]["base"]
                # On day 10, execute the big Melon harvest liquidation!
                if item == "MELON" and day in (10, 11) and qty > 0:
                    market_orders.append(["SELL", "MELON", min(10, qty)])
                    continue

                if p >= max(2, int(base_p * 0.18)):
                    sell_amt = min(qty, int(self.config.get("mid_sell_batch", 8)))
                    market_orders.append(["SELL", item, sell_amt])

        # Wheat selling (only surplus above 25 feed reserve)
        cows_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "COW")
        sheep_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "SHEEP")
        geese_now = sum(1 for r in farm.get("tiles", []) for t in r if isinstance(t, dict) and t.get("animal") == "GOOSE")
        total_anim = cows_now + sheep_now + geese_now

        wheat_in_shed = shed.get("WHEAT", 0)
        feed_reserve = max(24, total_anim)
        if wheat_in_shed > (feed_reserve + 8) and len(market_orders) < 8:
            market_orders.append(["SELL", "WHEAT", min(10, wheat_in_shed - feed_reserve)])

        # ----------------------------------------------------
        # 4. Feed Protection: Buy wheat if reserves dip
        # ----------------------------------------------------
        wheat_avail = wheat_in_shed + sum(inv.get("WHEAT", 0) for inv in inventories)
        if total_anim > 0 and wheat_avail < (total_anim + 2) and money >= 25 and len(market_orders) < 9:
            needed = min(8, (total_anim + 4) - wheat_avail)
            if needed > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", needed])

        # ----------------------------------------------------
        # 5. Daily Hiring (Fibonacci)
        # ----------------------------------------------------
        hires_today = int(farm.get("hires_today", 0))
        target_hands = self.get_max_hands(day)
        FIB_COSTS = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        hire_cost = FIB_COSTS[hires_today] if hires_today < len(FIB_COSTS) else 144
        if hires_today < target_hands and money >= hire_cost and len(market_orders) < 9:
            market_orders.append(["HIRE"])

        # ----------------------------------------------------
        # 6. 4-Quadrant Fast Expansion (NE Day 7, SW Day 9, SE Day 11)
        # ----------------------------------------------------
        if quads < 4 and len(market_orders) < 9:
            cost = next_land_cost(farm)
            if cost is not None:
                d_ne = int(self.config.get("expand_day_ne", 7))
                d_sw = int(self.config.get("expand_day_sw", 9))
                d_se = int(self.config.get("expand_day_se", 11))
                if quads == 1 and day >= d_ne and money >= cost:
                    market_orders.append(["BUY_LAND"])
                elif quads == 2 and day >= d_sw and money >= cost:
                    market_orders.append(["BUY_LAND"])
                elif quads == 3 and day >= d_se and money >= cost:
                    market_orders.append(["BUY_LAND"])

        # ----------------------------------------------------
        # 7. Mid-Game Animal Purchases (Scaling to 24 Animals)
        # ----------------------------------------------------
        if day >= 6 and day <= 22 and len(market_orders) < 9:
            cows_target = int(self.config.get("cow_target_mid", 7))
            sheep_target = int(self.config.get("sheep_target_mid", 6))
            goose_target = int(self.config.get("goose_target_mid", 11))

            held_cows = cows_now + shed.get("COW", 0) + sum(inv.get("COW", 0) for inv in inventories)
            held_sheep = sheep_now + shed.get("SHEEP", 0) + sum(inv.get("SHEEP", 0) for inv in inventories)
            held_goose = geese_now + shed.get("GOOSE", 0) + sum(inv.get("GOOSE", 0) for inv in inventories)

            # Buy Geese once NE is unlocked
            if quads >= 2 and held_goose < goose_target and money >= 400:
                market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
            elif quads >= 2 and held_cows < cows_target and money >= 500:
                market_orders.append(["BUY_ANIMAL", "COW", 1])
            elif quads >= 3 and held_sheep < sheep_target and money >= 600:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 1])

        # ----------------------------------------------------
        # 8. Seed Purchases (Wheat Flanks, Strawberries, Carrots, Tomatoes)
        # ----------------------------------------------------
        if day < 28 and len(market_orders) < 9:
            wheat_count = count_crop(farm, "WHEAT") + seeds.get("WHEAT", 0)
            straw_count = count_crop(farm, "STRAWBERRY") + seeds.get("STRAWBERRY", 0)
            carrot_count = count_crop(farm, "CARROT") + seeds.get("CARROT", 0)
            tomato_count = count_crop(farm, "TOMATO") + seeds.get("TOMATO", 0)

            # Target wheat based on unlocked quadrants
            target_wheat = 14 if quads == 1 else (20 if quads == 2 else 35)
            if wheat_count < target_wheat and money >= 50:
                market_orders.append(["BUY_SEED", "WHEAT", min(4, target_wheat - wheat_count)])
            elif quads >= 2 and straw_count < 22 and money >= 120:
                market_orders.append(["BUY_SEED", "STRAWBERRY", min(4, 22 - straw_count)])
            elif quads >= 3 and carrot_count < 18 and money >= 50:
                market_orders.append(["BUY_SEED", "CARROT", min(4, 18 - carrot_count)])
            elif quads >= 3 and tomato_count < 6 and money >= 70:
                market_orders.append(["BUY_SEED", "TOMATO", min(2, 6 - tomato_count)])

        return market_orders[:10]

    def plan_units(self, obs: Dict[str, Any], farm: Dict[str, Any]) -> Tuple[List[Any], List[List[Any]]]:
        farmer_pos = tuple(farm["farmer"])
        hands = farm.get("hands", []) or []
        all_units = [farmer_pos] + [tuple(h) for h in hands]
        inventories = obs.get("private", {}).get("inventories", [{} for _ in all_units])
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        step = int(obs.get("step", 0))
        shed = obs.get("private", {}).get("shed", {})
        seeds = obs.get("private", {}).get("seeds", {})
        unlocked_quads = farm.get("unlocked_quadrants", ["NW"])
        quads = len(unlocked_quads)
        tiles = farm.get("tiles", [])

        actions: List[List[Any]] = [["PASS"] for _ in all_units]
        claimed_targets = set()

        # Build structure lookups
        existing_pastures = set()
        existing_coops = set()
        occupied_pastures = set()
        occupied_coops = set()

        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict):
                    k = tile.get("kind")
                    if k == "PASTURE":
                        existing_pastures.add((x, y))
                        if tile.get("animal"):
                            occupied_pastures.add((x, y))
                    elif k == "COOP":
                        existing_coops.add((x, y))
                        if tile.get("animal"):
                            occupied_coops.add((x, y))

        # ----------------------------------------------------
        # Pass 1: IMMEDIATE CURRENT TILE EXECUTION
        # ----------------------------------------------------
        for u_idx, pos in enumerate(all_units):
            inv = inventories[u_idx] if u_idx < len(inventories) else {}
            tile = tiles[pos[1]][pos[0]] if 0 <= pos[1] < 10 and 0 <= pos[0] < 10 else None

            # 1. At Shed Actions
            if pos in SHED_ADJACENT:
                # Deposit carried non-feed items into shed
                carried_produce = sum(inv.get(prod, 0) for prod in ["WOOL", "MILK", "EGG", "MELON", "STRAWBERRY", "CARROT", "TOMATO"])
                carried_excess_fert = max(0, inv.get("FERTILIZER", 0) - 2)
                if carried_produce > 0 or carried_excess_fert > 0:
                    actions[u_idx] = ["DROP"]
                    continue

                # Pickup unplaced animals if shed has stock and empty structure exists
                if inv.get("COW", 0) == 0 and inv.get("SHEEP", 0) == 0 and inv.get("GOOSE", 0) == 0:
                    if shed.get("SHEEP", 0) > 0 and len(existing_pastures - occupied_pastures) > 0:
                        actions[u_idx] = ["PICKUP", "SHEEP", 1]
                        shed["SHEEP"] -= 1
                        continue
                    elif shed.get("COW", 0) > 0 and len(existing_pastures - occupied_pastures) > 0:
                        actions[u_idx] = ["PICKUP", "COW", 1]
                        shed["COW"] -= 1
                        continue
                    elif shed.get("GOOSE", 0) > 0 and len(existing_coops - occupied_coops) > 0:
                        actions[u_idx] = ["PICKUP", "GOOSE", 1]
                        shed["GOOSE"] -= 1
                        continue

                # Pickup wheat feed if low
                if inv.get("WHEAT", 0) <= 1 and shed.get("WHEAT", 0) > 0:
                    pick_qty = min(4, shed.get("WHEAT", 0))
                    actions[u_idx] = ["PICKUP", "WHEAT", pick_qty]
                    shed["WHEAT"] -= pick_qty
                    continue

            # 2. Structure Build Actions
            if tile is None:
                # If standing on a planned animal coordinate, BUILD IT!
                if pos in PASTURE_SPECS and pos not in existing_pastures and is_tile_unlocked(pos, unlocked_quads):
                    actions[u_idx] = ["BUILD_PASTURE"]
                    existing_pastures.add(pos)
                    continue
                elif pos in COOP_SPECS and pos not in existing_coops and is_tile_unlocked(pos, unlocked_quads):
                    actions[u_idx] = ["BUILD_COOP"]
                    existing_coops.add(pos)
                    continue

            # 3. Animal Maintenance on Current Tile
            if isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP"):
                # Place carried animal
                if not tile.get("animal"):
                    if tile.get("kind") == "PASTURE":
                        if inv.get("COW", 0) > 0:
                            actions[u_idx] = ["PLACE", "COW", 1]
                            inv["COW"] -= 1
                            continue
                        elif inv.get("SHEEP", 0) > 0:
                            actions[u_idx] = ["PLACE", "SHEEP", 1]
                            inv["SHEEP"] -= 1
                            continue
                    elif tile.get("kind") == "COOP":
                        if inv.get("GOOSE", 0) > 0:
                            actions[u_idx] = ["PLACE", "GOOSE", 1]
                            inv["GOOSE"] -= 1
                            continue

                # Feed animal
                if tile.get("animal") and not tile.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                    actions[u_idx] = ["FEED"]
                    inv["WHEAT"] -= 1
                    continue

                # Collect fertilizer
                if tile.get("fertilizer_available", False):
                    actions[u_idx] = ["COLLECT_FERTILIZER"]
                    inv["FERTILIZER"] = inv.get("FERTILIZER", 0) + 1
                    continue

                # Harvest animal product (milk, wool, eggs)
                if tile.get("yield_units", 0) > 0:
                    actions[u_idx] = ["HARVEST"]
                    continue

                # Care for animal
                if tile.get("animal") and not tile.get("cared_today", False):
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

                # Fertilize living crop if unit carries fertilizer
                fertilized_until = tile.get("fertilized_until_day", -1)
                if inv.get("FERTILIZER", 0) > 0 and fertilized_until <= day and day < 27:
                    actions[u_idx] = ["FERTILIZE"]
                    inv["FERTILIZER"] -= 1
                    continue

            # 5. Plant Seeds on Current Empty Tile
            if tile is None and pos not in ALL_ANIMAL_TILES and is_tile_unlocked(pos, unlocked_quads):
                # Choose crop by zone
                x, y = pos
                zone_crop = None
                if y <= 1:
                    zone_crop = "STRAWBERRY"
                elif y >= 7 and x <= 2:
                    zone_crop = "TOMATO"
                elif y >= 7 and x >= 3:
                    zone_crop = "CARROT"
                else:
                    zone_crop = "WHEAT"

                # Check melon override on Days 0-2
                if day <= 2 and seeds.get("MELON", 0) > 0 and (x in (1, 2) or y in (1, 2)):
                    zone_crop = "MELON"

                if zone_crop and seeds.get(zone_crop, 0) > 0:
                    actions[u_idx] = ["PLANT", zone_crop]
                    seeds[zone_crop] -= 1
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

        # Find candidate targets
        # Target A: Emergency watering (consecutive_unwatered >= 1)
        # Target B: Unfed animals (if unit carries wheat)
        # Target C: Uncollected fertilizer
        # Target D: Ready harvest
        # Target E: Unbuilt animal structures
        # Target F: Unwatered plants
        # Target G: Empty planting tiles
        # Target H: Shed drop (if carrying goods)

        for u_idx in unassigned:
            pos = all_units[u_idx]
            inv = inventories[u_idx] if u_idx < len(inventories) else {}

            best_target = None
            best_dist = 999

            # 1. Drop to shed if carrying items
            carried_items = sum(inv.get(p, 0) for p in ["WOOL", "MILK", "EGG", "MELON", "STRAWBERRY", "CARROT", "TOMATO", "FERTILIZER"])
            if carried_items >= 4 or (carried_items > 0 and inv.get("WHEAT", 0) == 0):
                target_shed = min(SHED_ADJACENT, key=lambda s: distance(pos, s))
                actions[u_idx] = next_step_toward(pos, target_shed)
                continue

            # 2. Emergency plant watering
            emergency_waters = []
            for px, py, pcrop in plant_tiles(farm):
                if (px, py) in claimed_targets:
                    continue
                ptile = tiles[py][px]
                if not ptile.get("watered_today", False) and ptile.get("consecutive_unwatered", 0) >= 1:
                    emergency_waters.append((distance(pos, (px, py)), (px, py)))
            if emergency_waters:
                emergency_waters.sort()
                d_w, w_pos = emergency_waters[0]
                claimed_targets.add(w_pos)
                actions[u_idx] = next_step_toward(pos, w_pos)
                continue

            # 3. Unfed animals (if carrying wheat)
            if inv.get("WHEAT", 0) > 0:
                unfed_anim = []
                for ax, ay in ALL_ANIMAL_TILES:
                    if (ax, ay) in claimed_targets:
                        continue
                    if 0 <= ay < 10 and 0 <= ax < 10 and isinstance(tiles[ay][ax], dict):
                        at = tiles[ay][ax]
                        if at.get("animal") and not at.get("fed_today", False):
                            unfed_anim.append((distance(pos, (ax, ay)), (ax, ay)))
                if unfed_anim:
                    unfed_anim.sort()
                    d_a, a_pos = unfed_anim[0]
                    claimed_targets.add(a_pos)
                    actions[u_idx] = next_step_toward(pos, a_pos)
                    continue

            # 4. Animal fertilizer collection
            fert_anim = []
            for ax, ay in ALL_ANIMAL_TILES:
                if (ax, ay) in claimed_targets:
                    continue
                if 0 <= ay < 10 and 0 <= ax < 10 and isinstance(tiles[ay][ax], dict):
                    at = tiles[ay][ax]
                    if at.get("fertilizer_available", False):
                        fert_anim.append((distance(pos, (ax, ay)), (ax, ay)))
            if fert_anim:
                fert_anim.sort()
                d_f, f_pos = fert_anim[0]
                claimed_targets.add(f_pos)
                actions[u_idx] = next_step_toward(pos, f_pos)
                continue

            # 5. Planned animal structure construction
            empty_struct_targets = []
            for sx, sy in (PASTURE_SPECS if quads < 2 else (PASTURE_SPECS + COOP_SPECS)):
                if (sx, sy) in claimed_targets:
                    continue
                if is_tile_unlocked((sx, sy), unlocked_quads):
                    st = tiles[sy][sx]
                    if st is None:
                        empty_struct_targets.append((distance(pos, (sx, sy)), (sx, sy)))
            if empty_struct_targets:
                empty_struct_targets.sort()
                d_s, s_pos = empty_struct_targets[0]
                claimed_targets.add(s_pos)
                actions[u_idx] = next_step_toward(pos, s_pos)
                continue

            # 6. Harvest ready crops
            harvest_t = find_harvest_target(farm, day, pos)
            if harvest_t is not None and harvest_t not in claimed_targets:
                claimed_targets.add(harvest_t)
                actions[u_idx] = next_step_toward(pos, harvest_t)
                continue

            # 7. Normal crop watering
            normal_waters = []
            for px, py, pcrop in plant_tiles(farm):
                if (px, py) in claimed_targets:
                    continue
                ptile = tiles[py][px]
                if not ptile.get("watered_today", False):
                    normal_waters.append((distance(pos, (px, py)), (px, py)))
            if normal_waters:
                normal_waters.sort()
                d_nw, nw_pos = normal_waters[0]
                claimed_targets.add(nw_pos)
                actions[u_idx] = next_step_toward(pos, nw_pos)
                continue

            # 8. Plant available seeds
            if sum(seeds.values()) > 0:
                empty_plant_tiles = [
                    p for p in unlocked_empty_tiles(farm)
                    if p not in ALL_ANIMAL_TILES and p not in claimed_targets
                ]
                if empty_plant_tiles:
                    near_plant = min(empty_plant_tiles, key=lambda p: distance(pos, p))
                    claimed_targets.add(near_plant)
                    actions[u_idx] = next_step_toward(pos, near_plant)
                    continue

            # 9. Walk to shed if carrying wheat or idle
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


_agent = TopMetaPolicy()

def agent(obs: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    return _agent(obs, *args, **kwargs)
