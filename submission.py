
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Kaggriculture submission.py
#
# A modular heuristic agent covering the basic game loop:
#   - survival / daily watering
#   - harvesting
#   - crop selection
#   - town-shop demand
#   - market selling
#   - seed purchasing
#   - farm expansion
#   - farm-hand hiring
#   - simple movement / field routing
#
# The agent deliberately keeps the game facts separate from
# tunable policy parameters so these values can later be evolved.
# ============================================================


# ------------------------------------------------------------
# Evolved / tunable configuration
# ------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    # Economy
    "cash_reserve": 500,
    "land_reserve": 1000,
    "max_hands": 3,
    "min_cash_for_hand": 800,

    # Market
    "sell_price_ratio": 0.80,
    "sell_min_price": 1,
    "sell_batch_size": 10,

    # Town demand
    "town_demand_weight": 1.00,
    "town_price_weight": 0.30,
    "town_scarcity_weight": 0.20,

    # Crop selection
    "wheat_ratio": 1.00,
    "carrot_ratio": 1.00,
    "tomato_ratio": 1.00,
    "strawberry_ratio": 1.00,
    "melon_ratio": 0.80,

    # Crop timing
    "harvest_buffer_days": 0,

    # Land
    "buy_land_enabled": True,
    "buy_land_cash_multiplier": 1.50,

    # Hiring
    "hire_enabled": True,

    # Optional livestock policy. Disabled in the manual baseline until tested.
    "max_animals": 0,
    "animal_species": "SHEEP",
    "sheep_goal": 0,
    "cow_goal": 0,

    # Navigation
    "prefer_nearest_action": True,
}


def set_policy_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a policy configuration, retaining defaults for omitted keys.

    Offline tools call this before an episode. The competition entry point
    remains usable on its own with the manually designed defaults above.
    """
    overrides = dict(config)
    CONFIG.clear()
    CONFIG.update(DEFAULT_CONFIG)
    CONFIG.update(overrides)
    return CONFIG


DEFAULT_CONFIG = dict(CONFIG)


# ------------------------------------------------------------
# Game facts
# ------------------------------------------------------------

CROP_INFO = {
    "WHEAT": {
        "seed_cost": 10,
        "base_price": 25,
        "first_yield_day": 2,
        "max_yield_day": 4,
    },
    "CARROT": {
        "seed_cost": 20,
        "base_price": 35,
        "first_yield_day": 2,
        "max_yield_day": 3,
    },
    "TOMATO": {
        "seed_cost": 50,
        "base_price": 60,
        "first_yield_day": 8,
        "max_yield_day": 11,
    },
    "STRAWBERRY": {
        "seed_cost": 100,
        "base_price": 120,
        "first_yield_day": 10,
        "max_yield_day": 16,
    },
    "MELON": {
        "seed_cost": 80,
        "base_price": 250,
        "first_yield_day": 10,
        "max_yield_day": 10,
    },
}

CROP_ORDER = [
    "WHEAT",
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
]

# Product -> animal producing it.
ANIMAL_PRODUCTS = {
    "EGG": "GOOSE",
    "MILK": "COW",
    "WOOL": "SHEEP",
}

ANIMAL_INFO = {
    "GOOSE": {"cost": 300, "structure": "COOP", "product": "EGG"},
    "COW": {"cost": 400, "structure": "PASTURE", "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "product": "WOOL"},
}

SHOP_DEMAND = {
    "BAKERY": {
        "EGG": 1,
        "WHEAT": 1,
    },
    "PIZZA_SHOP": {
        "MILK": 1,
        "TOMATO": 1,
        "WHEAT": 1,
    },
    "BRUNCH_SPOT": {
        "EGG": 1,
        "WHEAT": 1,
        "STRAWBERRY": 1,
    },
    "YARN_STORE": {
        "WOOL": 2,
    },
    "ICE_CREAM_SHOP": {
        "STRAWBERRY": 1,
        "MILK": 1,
        "WHEAT": 1,
    },
    "PET_CAFE": {
        "CARROT": 2,
    },
    "SMOOTHIE_SHOP": {
        "STRAWBERRY": 1,
        "MILK": 1,
    },
    "FARMERS_MARKET": {
        "WHEAT": 1,
        "CARROT": 1,
        "TOMATO": 1,
        "STRAWBERRY": 1,
    },
}

LAND_COSTS = [1000, 2000, 4000]

QUADRANT_ORDER = ["NW", "NE", "SW", "SE"]

# Default map is 10x10. The implementation derives center/land
# boundaries from board size where possible.
QUADRANT_ORIGINS = {
    "NW": (0, 0),
    "NE": (5, 0),
    "SW": (0, 5),
    "SE": (5, 5),
}


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def action(
    farmer: List[Any] | None = None,
    hands: Optional[List[List[Any]]] = None,
    market: Optional[List[List[Any]]] = None,
) -> Dict[str, Any]:
    return {
        "farmer": farmer or ["PASS"],
        "hands": hands or [],
        "market": market or [],
    }


def tile_kind(tile: Any) -> Optional[str]:
    if isinstance(tile, dict):
        return tile.get("kind")
    return None


def crop_ratio(crop: str) -> float:
    return float(CONFIG.get(f"{crop.lower()}_ratio", 1.0))


def distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def next_step_toward(
    current: Tuple[int, int],
    target: Tuple[int, int],
) -> List[Any]:
    """
    Move one square toward target.

    Locked tiles are passable, so we don't need a full pathfinder for
    the default rectangular board.
    """
    x, y = current
    tx, ty = target

    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]

    return ["PASS"]


def unlocked_empty_tiles(farm: Dict[str, Any]) -> List[Tuple[int, int]]:
    tiles = farm.get("tiles", [])
    result = []

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if tile is None:
                result.append((x, y))

    return result


def plant_tiles(
    farm: Dict[str, Any],
    crop: Optional[str] = None,
) -> List[Tuple[int, int, str]]:
    result = []

    tiles = farm.get("tiles", [])

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue

            if tile.get("kind") != "PLANT":
                continue

            if crop is not None and tile.get("crop") != crop:
                continue

            result.append((x, y, tile.get("crop", "")))

    return result


# ------------------------------------------------------------
# Town demand
# ------------------------------------------------------------

def town_product_demand(obs: Dict[str, Any]) -> Counter:
    shops = obs.get("town", {}).get("unlocked_shops", [])
    demand = Counter()

    for shop in shops:
        for product, amount in SHOP_DEMAND.get(shop, {}).items():
            demand[product] += amount

    return demand


def crop_demand_score(
    obs: Dict[str, Any],
    crop: str,
) -> float:
    demand = town_product_demand(obs)

    raw = float(demand.get(crop, 0))

    if not raw:
        return 0.0

    all_crop_demands = [
        float(demand.get(c, 0))
        for c in CROP_ORDER
    ]

    max_demand = max(all_crop_demands + [1.0])

    demand_component = raw / max_demand

    market = obs.get("market", {})
    prices = market.get("prices", {})
    inventory = market.get("inventory", {})

    base = CROP_INFO[crop]["base_price"]
    current_price = prices.get(crop, base)

    price_component = current_price / base

    inv = inventory.get(crop, 10000)
    scarcity_component = max(
        0.0,
        min(1.0, (10000 - inv) / 10000.0),
    )

    return (
        CONFIG["town_demand_weight"] * demand_component
        + CONFIG["town_price_weight"] * price_component
        + CONFIG["town_scarcity_weight"] * scarcity_component
    )


def choose_crop(obs: Dict[str, Any]) -> str:
    """
    Choose the next crop using:
      1. town demand
      2. current market price
      3. simple crop economics
      4. configurable crop ratios
    """

    market = obs.get("market", {})
    prices = market.get("prices", {})

    scores: Dict[str, float] = {}

    for crop in CROP_ORDER:
        info = CROP_INFO[crop]
        base = info["base_price"]

        market_ratio = prices.get(crop, base) / base

        # Basic profitability proxy. This is deliberately not a
        # replacement for a future learned/evolved crop model.
        profit_proxy = max(
            0.0,
            (prices.get(crop, base) - info["seed_cost"])
            / max(info["first_yield_day"], 1),
        )

        town = crop_demand_score(obs, crop)

        # Normalize profit enough that town demand can still matter.
        profit_component = min(
            2.0,
            profit_proxy / 30.0,
        )

        scores[crop] = (
            town
            + 0.25 * market_ratio
            + 0.20 * profit_component
        ) * crop_ratio(crop)

    return max(scores, key=scores.get)


# ------------------------------------------------------------
# Farm state
# ------------------------------------------------------------

def count_crop(farm: Dict[str, Any], crop: str) -> int:
    return sum(
        1
        for _, _, planted_crop in plant_tiles(farm, crop)
        if planted_crop == crop
    )


def count_empty_tiles(farm: Dict[str, Any]) -> int:
    return len(unlocked_empty_tiles(farm))


def find_priority_plant_action(
    farm: Dict[str, Any],
    crop: str,
) -> Optional[Tuple[int, int]]:
    """
    Prefer an empty tile in an already-unlocked quadrant.

    The observation marks locked tiles explicitly, so this is safe.
    """

    tiles = farm.get("tiles", [])

    for y, row in enumerate(tiles):
        for x, tile in enumerate(row):
            if tile is None:
                return (x, y)

    return None


def find_nearest_plant(
    farm: Dict[str, Any],
    position: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    empty = unlocked_empty_tiles(farm)

    if not empty:
        return None

    return min(
        empty,
        key=lambda p: distance(position, p),
    )


def animal_counts(farm: Dict[str, Any], species: str) -> Tuple[int, int]:
    """Return counts of occupied and empty matching structures."""
    structure = ANIMAL_INFO[species]["structure"]
    occupied = empty = 0
    for row in farm.get("tiles", []):
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == structure:
                if tile.get("animal"):
                    occupied += 1
                else:
                    empty += 1
    return occupied, empty


def animal_goals(config: Dict[str, Any]) -> Dict[str, int]:
    """Read mixed sheep/cow goals, or the older single-species setting."""
    mixed = {"SHEEP": int(config.get("sheep_goal", 0)),
             "COW": int(config.get("cow_goal", 0))}
    mixed = {species: max(0, count) for species, count in mixed.items() if count > 0}
    if mixed:
        return mixed
    species = config.get("animal_species", "SHEEP")
    count = int(config.get("max_animals", 0))
    return {species: count} if species in ANIMAL_INFO and count > 0 else {}


def animal_hand_actions(obs: Dict[str, Any], farm: Dict[str, Any]) -> List[Optional[List[Any]]]:
    """Assign distinct animal jobs to hands, respecting each hand's inventory."""
    goals = animal_goals(CONFIG)
    hands = farm.get("hands", []) or []
    if not goals:
        return [None for _ in hands]

    private = obs.get("private", {}) or {}
    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", []) or []
    hand_inventories = [inventories[i + 1] if i + 1 < len(inventories) else {}
                        for i in range(len(hands))]
    positions = [tuple(position) for position in hands]
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
                    jobs.append({"priority": 0, "target": (x, y),
                                 "action": ["FEED"], "needs": "WHEAT"})
                if tile.get("yield_units", 0) > 0:
                    jobs.append({"priority": 1, "target": (x, y), "action": ["HARVEST"]})
                if tile.get("fertilizer_available", False):
                    jobs.append({"priority": 2, "target": (x, y),
                                 "action": ["COLLECT_FERTILIZER"]})
                if not tile.get("cared_today", False):
                    jobs.append({"priority": 3, "target": (x, y), "action": ["CARE"]})
            elif tile.get("kind") in {ANIMAL_INFO[s]["structure"] for s in goals} and not tile.get("animal"):
                empty_structures.setdefault(tile["kind"], []).append((x, y))

    board_size = len(farm.get("tiles", []))
    half = board_size // 2
    shed_positions = [(half - 1, half - 1), (half, half - 1),
                      (half - 1, half), (half, half)]

    held_wheat = sum(int(inv.get("WHEAT", 0)) for inv in hand_inventories)
    wheat_needed = sum(not tile.get("fed_today", False) for _, _, tile in animals)
    wheat_to_pick = min(int(shed.get("WHEAT", 0)), max(0, wheat_needed - held_wheat))
    for position in shed_positions:
        if wheat_to_pick <= 0:
            break
        amount = min(5, wheat_to_pick)
        jobs.append({"priority": 0, "target": position,
                     "action": ["PICKUP", "WHEAT", amount]})
        wheat_to_pick -= amount

    carried_animals = sum(int(inv.get(species, 0)) for inv in inventories
                          for species in goals)
    available_places = sum(len(tiles) for tiles in empty_structures.values())
    stock_left_to_pick = max(0, available_places - carried_animals)
    place_slots = [position for structure in empty_structures.values() for position in structure]
    for species in goals:
        stock = int(shed.get(species, 0))
        animals_to_pick = min(stock, stock_left_to_pick)
        for index in range(animals_to_pick):
            jobs.append({"priority": 4, "target": shed_positions[index % len(shed_positions)],
                         "action": ["PICKUP", species, 1]})
        stock_left_to_pick -= animals_to_pick

    slot_index = 0
    for species in goals:
        carried = sum(int(inv.get(species, 0)) for inv in inventories)
        while carried > 0 and slot_index < len(place_slots):
            position = place_slots[slot_index]
            jobs.append({"priority": 4, "target": position,
                         "action": ["PLACE", species], "needs": species})
            slot_index += 1
            carried -= 1

    assigned: List[Optional[List[Any]]] = []
    for position, inventory in zip(positions, hand_inventories):
        eligible = [job for job in jobs
                    if not job.get("needs") or inventory.get(job["needs"], 0) > 0]
        if not eligible:
            assigned.append(None)
            continue
        job = min(eligible, key=lambda item: (
            item["priority"], distance(position, item["target"])))
        jobs.remove(job)
        assigned.append(job["action"] if position == job["target"]
                        else next_step_toward(position, job["target"]))
    return assigned


# ------------------------------------------------------------
# Crop lifecycle
# ------------------------------------------------------------

def find_urgent_water_target(
    farm: Dict[str, Any],
    position: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    candidates = []

    for x, y, crop in plant_tiles(farm):
        tile = farm["tiles"][y][x]

        if not tile.get("watered_today", False):
            candidates.append(
                (
                    distance(position, (x, y)),
                    x,
                    y,
                )
            )

    if not candidates:
        return None

    candidates.sort()
    _, x, y = candidates[0]

    return x, y


def find_harvest_target(
    farm: Dict[str, Any],
    current_day: int,
    position: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    candidates = []

    for x, y, crop in plant_tiles(farm):
        tile = farm["tiles"][y][x]

        planted_day = tile.get("planted_day", current_day)
        age = current_day - planted_day

        first_yield = CROP_INFO.get(
            crop,
            {}
        ).get(
            "first_yield_day",
            999,
        )

        yield_units = tile.get("yield_units", 0)

        # The simulator rejects harvests before first_yield_day even when
        # one-time crops start with a positive yield_units value.
        ready = yield_units > 0 and age >= first_yield

        if ready:
            candidates.append(
                (
                    distance(position, (x, y)),
                    x,
                    y,
                )
            )

    if not candidates:
        return None

    candidates.sort()
    _, x, y = candidates[0]

    return x, y


# ------------------------------------------------------------
# Market / shed
# ------------------------------------------------------------

def sell_orders(obs: Dict[str, Any]) -> List[List[Any]]:
    """
    Sell profitable items from the shed.

    Seeds are separate from shed inventory and are never sold here.
    """

    shed = obs.get("private", {}).get("shed", {})
    prices = obs.get("market", {}).get("prices", {})
    farms = obs.get("farms", [])
    player = int(obs.get("player", 0))
    farm = farms[player] if player < len(farms) else {}
    animal_count = sum(
        1 for row in farm.get("tiles", []) for tile in row
        if isinstance(tile, dict) and tile.get("animal")
    )
    wheat_reserve = animal_count * 2

    orders: List[List[Any]] = []

    for item, quantity in shed.items():
        if quantity <= 0:
            continue

        if item == "WHEAT":
            quantity = max(0, int(quantity) - wheat_reserve)
            if quantity <= 0:
                continue

        price = prices.get(item, 0)

        # Don't sell unknown resources.
        if price <= 0:
            continue

        # Avoid selling at a disastrous floor unless it is explicitly
        # configured to do so.
        if price < CONFIG["sell_min_price"]:
            continue

        base = (
            CROP_INFO.get(item, {}).get("base_price")
            or {
                "EGG": 50,
                "MILK": 160,
                "WOOL": 200,
                "FERTILIZER": 100,
            }.get(item, price)
        )

        if price < base * CONFIG["sell_price_ratio"]:
            # Low price: hold inventory for now.
            continue

        batch = min(
            int(quantity),
            int(CONFIG["sell_batch_size"]),
        )

        if batch > 0:
            orders.append([
                "SELL",
                item,
                batch,
            ])

    return orders


# ------------------------------------------------------------
# Expansion
# ------------------------------------------------------------

def next_land_cost(farm: Dict[str, Any]) -> Optional[int]:
    unlocked = farm.get("unlocked_quadrants", [])

    count = len(unlocked)

    if count >= 4:
        return None

    return LAND_COSTS[count - 1] if count > 0 else None


def should_buy_land(
    obs: Dict[str, Any],
    farm: Dict[str, Any],
) -> bool:
    if not CONFIG["buy_land_enabled"]:
        return False

    cost = next_land_cost(farm)

    if cost is None:
        return False

    money = farm.get("money", 0)

    empty = count_empty_tiles(farm)

    # Expand before the farm is completely starved of planting space.
    # The multiplier is configurable/evolvable.
    threshold = max(
        CONFIG["land_reserve"],
        cost * CONFIG["buy_land_cash_multiplier"],
    )

    return (
        money >= threshold
        and empty <= 4
    )


# ------------------------------------------------------------
# Hiring
# ------------------------------------------------------------

def should_hire(farm: Dict[str, Any]) -> bool:
    if not CONFIG["hire_enabled"]:
        return False

    hires = int(farm.get("hires_today", 0))
    money = farm.get("money", 0)

    if hires >= CONFIG["max_hands"]:
        return False

    return money >= CONFIG["min_cash_for_hand"]


# ------------------------------------------------------------
# Navigation
# ------------------------------------------------------------

def nearest_target(
    position: Tuple[int, int],
    targets: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    if not targets:
        return None

    return min(
        targets,
        key=lambda p: distance(position, p),
    )


def farmer_basic_action(
    obs: Dict[str, Any],
) -> List[Any]:
    """
    Main farmer controller.

    Priority:
      1. harvest ready crops
      2. water plants
      3. plant selected crop
      4. move toward an empty tile
      5. PASS
    """

    player = obs["player"]
    farm = obs["farms"][player]
    day = obs.get("day", 0)

    fx, fy = farm["farmer"]
    position = (fx, fy)

    # --------------------------------------------------------
    # 1. Harvest
    # --------------------------------------------------------

    harvest = find_harvest_target(
        farm,
        day,
        position,
    )

    if harvest is not None:
        hx, hy = harvest

        if position == (hx, hy):
            return ["HARVEST"]

        return next_step_toward(
            position,
            harvest,
        )

    # --------------------------------------------------------
    # 2. Water
    # --------------------------------------------------------

    water = find_urgent_water_target(
        farm,
        position,
    )

    if water is not None:
        wx, wy = water

        if position == (wx, wy):
            return ["WATER"]

        return next_step_toward(
            position,
            water,
        )

    # --------------------------------------------------------
    # 3. Plant
    # --------------------------------------------------------

    crop = choose_crop(obs)

    seeds = obs["private"].get(
        "seeds",
        {},
    )

    empty_target = find_nearest_plant(
        farm,
        position,
    )

    if empty_target is not None:
        ex, ey = empty_target

        if position == (ex, ey) and seeds.get(crop, 0) > 0:
            return [
                "PLANT",
                crop,
            ]

        # If we have the seed, move to an empty tile.
        if seeds.get(crop, 0) > 0:
            return next_step_toward(
                position,
                empty_target,
            )

    # --------------------------------------------------------
    # 4. No immediate work
    # --------------------------------------------------------

    return ["PASS"]


# ------------------------------------------------------------
# Main agent
# ------------------------------------------------------------

class KaggricultureAgent:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = set_policy_config(config) if config is not None else CONFIG

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        player = obs["player"]
        farm = obs["farms"][player]
        money = farm.get("money", 0)

        market_orders: List[List[Any]] = []

        # ----------------------------------------------------
        # A. Sell profitable shed inventory
        # ----------------------------------------------------

        market_orders.extend(
            sell_orders(obs)
        )

        # ----------------------------------------------------
        # B. Expand when the farm is running out of space
        # ----------------------------------------------------

        if should_buy_land(obs, farm):
            market_orders.append([
                "BUY_LAND"
            ])

        # ----------------------------------------------------
        # C. Hire hands when affordable
        # ----------------------------------------------------

        if should_hire(farm):
            market_orders.append([
                "HIRE"
            ])

        # ----------------------------------------------------
        # D. Buy seeds for the crop the strategy currently wants
        # ----------------------------------------------------

        crop = choose_crop(obs)
        seeds = obs["private"].get(
            "seeds",
            {},
        )

        # Keep a small seed buffer rather than buying an unlimited
        # quantity every turn.
        desired_seed_buffer = 2

        seed_count = seeds.get(crop, 0)

        if seed_count < desired_seed_buffer:
            cost = CROP_INFO[crop]["seed_cost"]

            available_cash = (
                money
                - self.config["cash_reserve"]
            )

            affordable = max(
                0,
                int(available_cash // cost),
            )

            buy_count = min(
                desired_seed_buffer - seed_count,
                affordable,
                3,
            )

            if buy_count > 0:
                market_orders.append([
                    "BUY_SEED",
                    crop,
                    buy_count,
                ])

        # E. Buy livestock only for matching empty structures.
        goals = animal_goals(self.config)
        if goals:
            private = obs["private"]
            inventories = private.get("inventories", [])
            shed = private.get("shed", {})
            structures = {ANIMAL_INFO[s]["structure"] for s in goals}
            empty_structures = sum(
                1 for row in farm.get("tiles", []) for tile in row
                if isinstance(tile, dict) and tile.get("kind") in structures
                and not tile.get("animal")
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
            vacant_for_stock = max(0, empty_structures - stock_total)
            species_order = [s for s in ("COW", "SHEEP", "GOOSE") if s in goals]
            species_order.extend(s for s in goals if s not in species_order)
            for animal_species in species_order:
                deficit = goals[animal_species] - occupied_by_species[animal_species] - stock_by_species[animal_species]
                animal_price = ANIMAL_INFO[animal_species]["cost"]
                if (deficit > 0 and vacant_for_stock > 0
                        and money - self.config["cash_reserve"] >= animal_price):
                    market_orders.append(["BUY_ANIMAL", animal_species, 1])
                    break

            animal_n = sum(occupied_by_species.values())
            wheat_available = int(obs["private"].get("shed", {}).get("WHEAT", 0))
            wheat_available += sum(int(inv.get("WHEAT", 0)) for inv in inventories)
            wheat_target = animal_n * 2
            wheat_price = int(obs.get("market", {}).get("prices", {}).get("WHEAT", 25))
            wheat_affordable = max(0, int((money - self.config["cash_reserve"])
                                          // max(1, wheat_price)))
            wheat_buy = min(10, max(0, wheat_target - wheat_available), wheat_affordable)
            if animal_n and wheat_buy:
                market_orders.append(["BUY_PRODUCT", "WHEAT", wheat_buy])

        # ----------------------------------------------------
        # F. Main farmer action
        # ----------------------------------------------------

        farmer_action = farmer_basic_action(obs)

        if goals:
            private = obs["private"]
            structures = {ANIMAL_INFO[s]["structure"] for s in goals}
            structure_count = sum(
                1 for row in farm.get("tiles", []) for tile in row
                if isinstance(tile, dict) and tile.get("kind") in structures
            )
            stock = sum(int(private.get("shed", {}).get(s, 0))
                        + sum(int(inv.get(s, 0)) for inv in private.get("inventories", []))
                        for s in goals)
            if structure_count + stock < sum(goals.values()):
                farmer_position = tuple(farm["farmer"])
                empty = find_nearest_plant(farm, farmer_position)
                if empty is not None:
                    build_species = next((s for s in ("COW", "SHEEP", "GOOSE") if s in goals), next(iter(goals)))
                    farmer_action = (["BUILD_" + ANIMAL_INFO[build_species]["structure"]]
                                     if farmer_position == empty
                                     else next_step_toward(farmer_position, empty))

        # ----------------------------------------------------
        # F. Farm hands
        #
        # Hands start adjacent to the shed and can work on the
        # farm independently. For now we give each hand a useful
        # basic task based on its current location.
        # ----------------------------------------------------

        hand_actions: List[List[Any]] = []

        animal_actions = animal_hand_actions(obs, farm)
        for hand_index, hand in enumerate(farm.get("hands", [])):
            animal_action = animal_actions[hand_index] if hand_index < len(animal_actions) else None
            if animal_action is not None:
                hand_actions.append(animal_action)
                continue
            hx, hy = hand
            hand_position = (hx, hy)

            # Water nearest unwatered plant.
            water_target = find_urgent_water_target(
                farm,
                hand_position,
            )

            if water_target is not None:
                if hand_position == water_target:
                    hand_actions.append(["WATER"])
                else:
                    hand_actions.append(
                        next_step_toward(
                            hand_position,
                            water_target,
                        )
                    )
                continue

            # Otherwise harvest a ready crop.
            harvest_target = find_harvest_target(
                farm,
                obs.get("day", 0),
                hand_position,
            )

            if harvest_target is not None:
                if hand_position == harvest_target:
                    hand_actions.append(["HARVEST"])
                else:
                    hand_actions.append(
                        next_step_toward(
                            hand_position,
                            harvest_target,
                        )
                    )
                continue

            hand_actions.append(["PASS"])

        # Kaggriculture silently drops market orders beyond the
        # configured per-turn maximum. We cap defensively here.
        max_orders = 10

        return {
            "farmer": farmer_action,
            "hands": hand_actions,
            "market": market_orders[:max_orders],
        }


# ------------------------------------------------------------
# Competition entry point
# ------------------------------------------------------------

_agent = KaggricultureAgent(CONFIG)


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    return _agent(obs)


# ------------------------------------------------------------
# Optional local test
#
# Uncomment when running in an environment with
# kaggle-environments installed.
# ------------------------------------------------------------
#
# if __name__ == "__main__":
#     from kaggle_environments import make
#
#     env = make(
#         "kaggriculture",
#         configuration={
#             "episodeSteps": 720,
#             "seed": 1234,
#         },
#         debug=True,
#     )
#
#     env.run([agent, "random"])
#
#     final = env.steps[-1]
#     for i, state in enumerate(final):
#         print(
#             f"Player {i}: "
#             f"reward={state.reward}, "
#             f"status={state.status}"
#         )
