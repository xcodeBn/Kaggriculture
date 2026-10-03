from __future__ import annotations

import math
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

    # NOTE: town_demand_weight/town_price_weight/town_scarcity_weight and the
    # per-crop *_ratio keys were previously here but are not read by
    # choose_crop() (it uses hardcoded price thresholds instead) — they were
    # dead search-space dimensions. crop_demand_score()/crop_ratio() still
    # exist below if you want to wire them into crop selection later; until
    # then, leaving these keys out keeps the optimizer from wasting trials
    # on parameters with no effect.

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

# In the local Kaggriculture interpreter, the final agent observation for the
# default 720-step episode is step 718 (the interpreter completes immediately
# after processing that action).
FINAL_ACTION_STEP = 718

# Competition policy settings are embedded here so the submission is standalone.
# Offline experiments can promote a reviewed JSON config into this block.
COMPETITION_CONFIG: Dict[str, Any] = {
    'cash_reserve': 200,
    'land_reserve': 1500,
    'buy_land_cash_multiplier': 1.15,
    'buy_land_enabled': True,
    'expand_day_first': 11,
    'expand_day_second': 14,
    'max_animals': 12,
    'cow_goal': 6,
    'sheep_goal': 6,
    'animal_species': 'COW',
    'max_hands': 9,
    'min_cash_for_hand': 50,
    'sell_batch_size': 10,
    'sell_price_ratio': 0.25,
    'sell_min_price': 1,
    'harvest_buffer_days': 0,
    'hire_enabled': True,
    'prefer_nearest_action': True,
}


# ------------------------------------------------------------
# Closed-Form Market Model & Price Dynamics
# ------------------------------------------------------------

MARKET_PARAMS: Dict[str, Dict[str, Any]] = {
    "WHEAT": {"base": 25, "I0": 10000, "T": 400, "below_func": "sqrt", "below_target": 0.80, "above_func": "log", "above_target": 0.20},
    "CARROT": {"base": 35, "I0": 10000, "T": 450, "below_func": "hinge", "below_target": 1.00, "above_func": "sqrt", "above_target": 0.70},
    "TOMATO": {"base": 60, "I0": 10000, "T": 200, "below_func": "hinge", "below_target": 0.40, "above_func": "sqrt", "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": 10000, "T": 100, "below_func": "sqrt", "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON": {"base": 250, "I0": 10000, "T": 300, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.60},
    "EGG": {"base": 50, "I0": 10000, "T": 332, "below_func": "hinge", "below_target": 0.40, "above_func": "log", "above_target": 0.20},
    "MILK": {"base": 160, "I0": 10000, "T": 122, "below_func": "sqrt", "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL": {"base": 200, "I0": 10000, "T": 105, "below_func": "log", "below_target": 0.20, "above_func": "sq", "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": 10000, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}


def _eval_market_func(func_name: str, x: float, T: float) -> float:
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
    return float(x)


def calculate_price(item: str, inv: float) -> int:
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
        sign = 1.0
        func_name = params["below_func"]
        target = params["below_target"]
        x = -diff
    else:
        sign = -1.0
        func_name = params["above_func"]
        target = params["above_target"]
        x = diff
    f_T = _eval_market_func(func_name, T, T)
    amp = target * base / f_T if f_T > 0 else 0.0
    raw_price = base + sign * amp * _eval_market_func(func_name, x, T)
    return max(1, round(raw_price))


def estimate_inventory_from_price(item: str, observed_price: int, hint_above: bool = True) -> int:
    params = MARKET_PARAMS.get(item)
    if not params:
        return 10000
    I0 = params["I0"]
    base = params["base"]
    T = params["T"]
    if observed_price == base:
        return I0
    if observed_price > base or (not hint_above and observed_price >= base):
        func_name = params["below_func"]
        target = params["below_target"]
        f_T = _eval_market_func(func_name, T, T)
        amp = target * base / f_T if f_T > 0 else 1.0
        needed_fx = (observed_price - base) / amp
        low, high = 0.0, float(I0)
        for _ in range(30):
            mid = (low + high) / 2.0
            if _eval_market_func(func_name, mid, T) < needed_fx:
                low = mid
            else:
                high = mid
        return max(0, int(round(I0 - low)))
    else:
        func_name = params["above_func"]
        target = params["above_target"]
        f_T = _eval_market_func(func_name, T, T)
        amp = target * base / f_T if f_T > 0 else 1.0
        needed_fx = (base - observed_price) / amp
        low, high = 0.0, 50000.0
        for _ in range(35):
            mid = (low + high) / 2.0
            if _eval_market_func(func_name, mid, T) < needed_fx:
                low = mid
            else:
                high = mid
        return int(round(I0 + low))


def optimal_sale_batch(item: str, shed_qty: int, start_inv: int, min_marginal_ratio: float = 0.55) -> int:
    if shed_qty <= 0:
        return 0
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


def find_weeds(farm: Dict[str, Any]) -> List[Tuple[int, int]]:
    """Return coordinates of all weeds on unlocked tiles."""
    weeds = []
    for y, row in enumerate(farm.get("tiles", [])):
        for x, tile in enumerate(row):
            if isinstance(tile, dict) and tile.get("kind") == "WEED":
                weeds.append((x, y))
    return weeds


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


def choose_crop(obs: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Choose the next crop using:
      0. livestock feed security (capped at 4 fields so cash crops dominate)
      1. remaining season growth horizon
      2. real-time market prices and gluts (pivoting away from saturated goods)
      3. town shop demand
    """
    config = cfg if cfg is not None else CONFIG
    day = int(obs.get("day", 0))

    # 1. Horizon checks: crops planted too late can never be harvested before season ends (step 720)
    if day >= 28:
        return None
    if day >= 26:
        return "WHEAT"
    if day >= 24:
        return "CARROT"

    player = obs.get("player")
    farms = obs.get("farms", [])
    farm = farms[player] if player is not None and player < len(farms) else {}

    prices = obs.get("market", {}).get("prices", {})
    wheat_price = prices.get("WHEAT", 25)
    milk_price = prices.get("MILK", 160)
    straw_price = prices.get("STRAWBERRY", 120)
    melon_price = prices.get("MELON", 250)
    carrot_price = prices.get("CARROT", 35)

    # 2. Animal Feed Security: maintain up to 4 wheat fields if livestock is profitable
    goals = animal_goals(config)
    total_animals = sum(goals.values())
    if total_animals > 0 and farm:
        current_wheat = count_crop(farm, "WHEAT")
        if current_wheat < 4 and (milk_price >= 40 or wheat_price <= 35):
            return "WHEAT"

    # 3. Market-Adaptive Selection based on real-time prices & farm crop density
    straw_cnt = count_crop(farm, "STRAWBERRY") if farm else 0
    melon_cnt = count_crop(farm, "MELON") if farm else 0

    # Premium pricing detection
    if straw_price >= 100 and straw_cnt < 28 and day <= 22:
        return "STRAWBERRY"
    if melon_price >= 180 and melon_cnt < 8 and day <= 16:
        return "MELON"
    if carrot_price >= 60 and day <= 25:
        return "CARROT"

    if day <= 16:
        if melon_price >= 160 and melon_cnt < 8:
            if straw_price < 75 or straw_cnt >= 20:
                return "MELON"
        if straw_price >= 55 and straw_cnt < 24:
            return "STRAWBERRY"
        if melon_cnt < 6 and melon_price >= 140:
            return "MELON"
        return "CARROT"
    elif day <= 23:
        if straw_price >= 65 and straw_cnt < 25 and day <= 20:
            return "STRAWBERRY"
        return "CARROT"
    else:
        return "CARROT"


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


def find_central_empty_tile(
    farm: Dict[str, Any],
    position: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """Find empty tile closest to central corridor (4.5, 4.5) for tight pasture clustering."""
    empty = unlocked_empty_tiles(farm)
    if not empty:
        return None
    return min(
        empty,
        key=lambda p: (abs(p[0] - 4.5) + abs(p[1] - 4.5), distance(position, p)),
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
            jobs.append({"priority": 1, "target": shed_positions[index % len(shed_positions)],
                         "action": ["PICKUP", species, 1]})
        stock_left_to_pick -= animals_to_pick

    slot_index = 0
    for species in goals:
        carried = sum(int(inv.get(species, 0)) for inv in inventories)
        while carried > 0 and slot_index < len(place_slots):
            position = place_slots[slot_index]
            jobs.append({"priority": 1, "target": position,
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
    # Kaggriculture's documented default season is 720 turns, with step
    # zero at the start. On the final turn there is no benefit to carrying
    # saleable shed inventory forward, so liquidate even below the normal
    # price threshold and release wheat reserved for animal feed.
    observed_step = obs.get("step")
    final_turn = observed_step is not None and int(observed_step) >= FINAL_ACTION_STEP
    stagger_phase = observed_step is not None and int(observed_step) >= (FINAL_ACTION_STEP - 48)

    inventories_mkt = obs.get("market", {}).get("inventory", {})
    unlocked_shops = obs.get("town", {}).get("unlocked_shops", [])
    drain_rates = {item: 1.0 for item in ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL"]}
    for sh in unlocked_shops:
        for it, qty_req in SHOP_DEMAND.get(sh, {}).items():
            drain_rates[it] = drain_rates.get(it, 0.0) + (qty_req * 6.0)

    animal_count = sum(
        1 for row in farm.get("tiles", []) for tile in row
        if isinstance(tile, dict) and tile.get("animal")
    )
    wheat_reserve = animal_count * 2

    orders: List[List[Any]] = []

    for item, quantity in shed.items():
        quantity = int(quantity)
        if quantity <= 0:
            continue

        if item == "WHEAT" and not final_turn:
            quantity = max(0, quantity - wheat_reserve)
            if quantity <= 0:
                continue

        price = prices.get(item, 0)
        if price <= 0:
            continue

        if price < CONFIG["sell_min_price"] and not final_turn:
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

        min_ratio = 0.15 if stagger_phase else CONFIG["sell_price_ratio"]
        if not final_turn and price < base * min_ratio:
            continue

        has_town_drain = drain_rates.get(item, 1.0) > 1.5
        inv = inventories_mkt.get(item)
        if inv is None:
            inv = estimate_inventory_from_price(item, price)

        if final_turn:
            batch = quantity
        elif stagger_phase:
            batch = min(quantity, 4 if item in {"MILK", "WOOL"} else 6)
        elif has_town_drain:
            batch = optimal_sale_batch(item, quantity, inv, min_marginal_ratio=0.55)
            batch = min(batch, 8 if item in {"MILK", "WOOL", "MELON"} else 12)
        else:
            batch = min(quantity, int(CONFIG["sell_batch_size"]))

        if batch > 0:
            orders.append([
                "SELL",
                item,
                batch,
            ])

    return orders


def final_turn_liquidation(obs: Dict[str, Any], farm: Dict[str, Any]) -> tuple[List[List[Any]], List[List[Any]]]:
    """Deposit carried saleable goods at the shed, then sell them all.

    Unit actions run before market orders. On the final default-season turn,
    hands already on shed-access tiles can therefore deposit their inventory and
    have it sold in the same turn. Movement to the shed would take too long.
    """
    private = obs.get("private", {})
    shed = dict(private.get("shed", {}))
    inventories = private.get("inventories", [])
    positions = [farm.get("farmer", [None, None]), *farm.get("hands", [])]
    final_actions: List[List[Any]] = []
    projected_total = sum(max(0, int(n)) for n in shed.values())
    shed_tiles = {(4, 4), (5, 4), (4, 5), (5, 5)}
    capacity = 100  # Kaggriculture's documented default shed capacity.
    prices = obs.get("market", {}).get("prices", {})

    for index, position in enumerate(positions):
        inventory = inventories[index] if index < len(inventories) else {}
        room = max(0, capacity - projected_total)
        if (len(position) != 2 or tuple(position) not in shed_tiles or room == 0):
            final_actions.append(["PASS"])
            continue
        saleable = [(int(prices.get(item, 0)) * min(max(0, int(quantity)), room),
                     item, min(max(0, int(quantity)), room))
                    for item, quantity in inventory.items()
                    if int(prices.get(item, 0)) > 0 and int(quantity) > 0]
        if not saleable:
            final_actions.append(["PASS"])
            continue
        _, item, quantity = max(saleable)
        # PLACE with an item and amount deposits only that saleable product;
        # DROP would also dump unsaleable animals into the limited-capacity shed.
        final_actions.append(["PLACE", item, quantity])
        shed[item] = int(shed.get(item, 0)) + quantity
        projected_total += quantity

    liquidation_obs = dict(obs)
    liquidation_obs["private"] = dict(private, shed=shed)
    return sell_orders(liquidation_obs), final_actions


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
    cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    config = cfg if cfg is not None else CONFIG
    if not config.get("buy_land_enabled", True):
        return False

    day = int(obs.get("day", 0))
    if day > 20:
        return False

    cost = next_land_cost(farm)
    if cost is None:
        return False

    quads = len(farm.get("unlocked_quadrants", []))
    if quads >= 3:
        return False

    money = farm.get("money", 0)
    empty = count_empty_tiles(farm)

    thresh = max(
        config.get("land_reserve", 1500),
        cost * config.get("buy_land_cash_multiplier", 1.15) + config.get("cash_reserve", 200),
    )

    first_day = config.get("expand_day_first", 11)
    second_day = config.get("expand_day_second", 14)

    if quads == 1 and day >= first_day and money >= thresh and empty <= 12:
        return True
    elif quads == 2 and day >= second_day and (money >= 6000 or (money >= thresh and empty <= 26)):
        return True

    return False


# ------------------------------------------------------------
# Hiring
# ------------------------------------------------------------

def should_hire(
    farm: Dict[str, Any],
    obs: Optional[Dict[str, Any]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> bool:
    config = cfg if cfg is not None else CONFIG
    if not config.get("hire_enabled", True):
        return False

    hires = int(farm.get("hires_today", 0))
    money = farm.get("money", 0)
    day = int(obs.get("day", 0)) if obs is not None else 0

    max_h = int(config.get("max_hands", 9))
    target_hands = min(max_h, 6 if day < 5 else (8 if day < 9 else (max_h if day < 28 else 6)))

    if hires >= target_hands:
        return False

    return money >= config.get("min_cash_for_hand", 50)


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
      0. dig weed on current tile
      1. harvest ready crops
      2. water plants
      3. plant selected crop
      4. clear nearest weed
      5. PASS
    """

    player = obs["player"]
    farm = obs["farms"][player]
    day = obs.get("day", 0)

    fx, fy = farm["farmer"]
    position = (fx, fy)

    # --------------------------------------------------------
    # 0. Dig weed on current tile
    # --------------------------------------------------------
    tile = farm["tiles"][fy][fx]
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]

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

    if empty_target is not None and crop is not None:
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
    # 4. Dig nearest weed on unlocked tiles
    # --------------------------------------------------------
    weeds = find_weeds(farm)
    if weeds:
        nearest_weed = min(weeds, key=lambda w: distance(position, w))
        if position == nearest_weed:
            return ["DIG"]
        return next_step_toward(position, nearest_weed)

    # --------------------------------------------------------
    # 5. No immediate work
    # --------------------------------------------------------

    return ["PASS"]


# ------------------------------------------------------------
# Main agent
# ------------------------------------------------------------

class KaggricultureAgent:

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = set_policy_config(config) if config is not None else CONFIG

    def _base_step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
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
        # B. Expand when the farm is ready for more land
        # ----------------------------------------------------

        if should_buy_land(obs, farm, self.config):
            market_orders.append([
                "BUY_LAND"
            ])

        # ----------------------------------------------------
        # C. Hire hands when affordable and needed
        # ----------------------------------------------------

        if should_hire(farm, obs, self.config):
            market_orders.append([
                "HIRE"
            ])

        # ----------------------------------------------------
        # D. Buy seeds for the crop the strategy currently wants
        # ----------------------------------------------------

        crop = choose_crop(obs, self.config)
        seeds = obs["private"].get(
            "seeds",
            {},
        )
        day = int(obs.get("day", 0))
        quads = len(farm.get("unlocked_quadrants", []))

        # Dynamic seed buffer: scale with unlocked quadrants and empty tiles,
        # but do not buy seeds past Day 28 since crops cannot mature before step 720.
        desired_seed_buffer = (
            min(18, max(4, count_empty_tiles(farm))) if quads > 1 else 4
        ) if (crop is not None and day < 28) else 0

        seed_count = seeds.get(crop, 0) if crop else 0

        if crop is not None and seed_count < desired_seed_buffer:
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
                8 if quads > 1 else 5,
            )

            if buy_count > 0:
                market_orders.append([
                    "BUY_SEED",
                    crop,
                    buy_count,
                ])

        # Also buy wheat seeds if feed security needs them
        goals = animal_goals(self.config)
        if goals and crop != "WHEAT" and day < 26:
            wheat_seeds = seeds.get("WHEAT", 0)
            if wheat_seeds < 3 and money - self.config["cash_reserve"] >= 30:
                market_orders.append(["BUY_SEED", "WHEAT", 3 - wheat_seeds])

        # E. Buy livestock only for matching empty structures (during investment phase).
        is_investment_phase = day <= 22
        prices = obs.get("market", {}).get("prices", {})
        milk_p = int(prices.get("MILK", 160))
        wool_p = int(prices.get("WOOL", 200))
        wheat_price = int(prices.get("WHEAT", 25))

        if goals and is_investment_phase:
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
                # Protective market check: do not buy animal if product price crashed
                if animal_species == "COW" and milk_p < 40:
                    continue
                if animal_species == "SHEEP" and wool_p < 40:
                    continue
                deficit = goals[animal_species] - occupied_by_species[animal_species] - stock_by_species[animal_species]
                animal_price = ANIMAL_INFO[animal_species]["cost"]
                if (deficit > 0 and vacant_for_stock > 0
                        and money - self.config["cash_reserve"] >= animal_price):
                    market_orders.append(["BUY_ANIMAL", animal_species, 1])
                    break

        # Emergency wheat buy if animals might starve - with price sanity check!
        if goals:
            private = obs["private"]
            inventories = private.get("inventories", [])
            shed = private.get("shed", {})
            occupied_by_species = {
                s: sum(1 for row in farm.get("tiles", []) for tile in row
                       if isinstance(tile, dict) and tile.get("animal") == s)
                for s in goals
            }
            animal_n = sum(occupied_by_species.values())
            wheat_available = int(shed.get("WHEAT", 0)) + sum(int(inv.get("WHEAT", 0)) for inv in inventories)
            if (animal_n > 0 and wheat_available < animal_n and money - self.config["cash_reserve"] >= wheat_price
                    and (milk_p >= 30 or wool_p >= 30 or wheat_price <= milk_p)):
                wheat_buy = min(10, (animal_n * 2) - wheat_available)
                if wheat_buy > 0:
                    market_orders.append(["BUY_PRODUCT", "WHEAT", wheat_buy])

        # ----------------------------------------------------
        # F. Main farmer action
        # ----------------------------------------------------

        farmer_action = farmer_basic_action(obs)

        if goals and is_investment_phase:
            structures = {ANIMAL_INFO[s]["structure"] for s in goals}
            structure_count = sum(
                1 for row in farm.get("tiles", []) for tile in row
                if isinstance(tile, dict) and tile.get("kind") in structures
            )
            if structure_count < sum(goals.values()):
                farmer_position = tuple(farm["farmer"])
                empty = find_central_empty_tile(farm, farmer_position)
                if empty is not None:
                    build_species = next((s for s in ("COW", "SHEEP", "GOOSE") if s in goals), next(iter(goals)))
                    farmer_action = (["BUILD_" + ANIMAL_INFO[build_species]["structure"]]
                                     if farmer_position == empty
                                     else next_step_toward(farmer_position, empty))

        # ----------------------------------------------------
        # F. Farm hands
        # ----------------------------------------------------

        hand_actions: List[List[Any]] = []
        weeds = find_weeds(farm)
        claimed_weeds: set = set()
        avail_seeds = seeds.get(crop, 0) if crop else 0
        claimed_empty: set = set()

        animal_actions = animal_hand_actions(obs, farm)
        for hand_index, hand in enumerate(farm.get("hands", [])):
            animal_action = animal_actions[hand_index] if hand_index < len(animal_actions) else None
            if animal_action is not None:
                hand_actions.append(animal_action)
                continue
            hx, hy = hand
            hand_position = (hx, hy)

            # Dig weed on standing tile
            h_tile = farm["tiles"][hy][hx]
            if isinstance(h_tile, dict) and h_tile.get("kind") == "WEED":
                hand_actions.append(["DIG"])
                claimed_weeds.add(hand_position)
                continue

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

            # Dig weed on unlocked tiles (at most 1 hand per weed)
            avail_weeds = [w for w in weeds if w not in claimed_weeds]
            if avail_weeds:
                nearest_w = min(avail_weeds, key=lambda w: distance(hand_position, w))
                claimed_weeds.add(nearest_w)
                if hand_position == nearest_w:
                    hand_actions.append(["DIG"])
                else:
                    hand_actions.append(next_step_toward(hand_position, nearest_w))
                continue

            hand_actions.append(["PASS"])

        # Liquidate instead of buying seeds, land, animals, or labor that
        # cannot pay back before the documented 720-turn season ends.
        observed_step = obs.get("step")
        if observed_step is not None and int(observed_step) >= FINAL_ACTION_STEP:
            liq_orders, final_actions = final_turn_liquidation(obs, farm)
            farmer_action = final_actions[0]
            hand_actions = final_actions[1:]
            market_orders = liq_orders[:10]

        # Kaggriculture silently drops market orders beyond the
        # configured per-turn maximum. We cap defensively here.
        max_orders = 10

        return {
            "farmer": farmer_action,
            "hands": hand_actions,
            "market": market_orders[:max_orders],
        }

    def _apply_shared_overrides(self, obs: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Land expansion, hiring, seed buffer, and idle-hand planting.

        This used to be duplicated (with drifted constants) in both
        KaggricultureAgent.__call__ and EvolvedAgent.__call__. It now runs
        once, always reading thresholds from self.config so every knob here
        is actually tunable by the offline search, and so a fix here can't
        silently miss the evolved subclass.
        """
        player = obs["player"]
        farm = obs["farms"][player]
        money = farm.get("money", 0)
        day = int(obs.get("day", 0))
        quads = len(farm.get("unlocked_quadrants", []))

        # Protect against a crashing livestock market (e.g. dairy baron opponent).
        prices = obs.get("market", {}).get("prices", {})
        milk_p = prices.get("MILK", 160)
        wool_p = prices.get("WOOL", 200)
        wheat_p = prices.get("WHEAT", 25)
        if milk_p < 30 and wool_p < 30 and wheat_p > milk_p:
            action["market"] = [m for m in action.get("market", []) if not (m and m[0] == "BUY_PRODUCT" and m[1] == "WHEAT")]
        if milk_p < 40:
            action["market"] = [m for m in action.get("market", []) if not (m and m[0] == "BUY_ANIMAL" and m[1] == "COW")]
        if wool_p < 40:
            action["market"] = [m for m in action.get("market", []) if not (m and m[0] == "BUY_ANIMAL" and m[1] == "SHEEP")]

        # Land expansion. Reuses should_buy_land so this can't drift from the
        # _base_step check by hardcoding a different multiplier/reserve.
        if should_buy_land(obs, farm, self.config):
            has_land = any(m and m[0] == "BUY_LAND" for m in action.get("market", []))
            if not has_land and len(action.get("market", [])) < 10:
                action["market"].append(["BUY_LAND"])

        # Hiring, via the same should_hire used in _base_step.
        if should_hire(farm, obs, self.config):
            has_hire = any(m and m[0] == "HIRE" for m in action.get("market", []))
            if not has_hire and len(action.get("market", [])) < 10:
                action["market"].append(["HIRE"])

        # Adaptive crop selection (single source of truth: choose_crop already
        # checks the day>=28 horizon cutoff before any feed-security logic).
        crop = choose_crop(obs, self.config)
        seeds = obs["private"].get("seeds", {})
        avail_seeds = seeds.get(crop, 0) if crop else 0

        # Maintain seed buffer (never buys seeds after day 28 - choose_crop
        # returns None past that point, which short-circuits this block).
        if crop is not None and day < 28:
            desired_seeds = min(18, max(4, count_empty_tiles(farm))) if quads > 1 else 4
            if seeds.get(crop, 0) < desired_seeds and money >= self.config.get("cash_reserve", 180) + 80:
                cost_s = CROP_INFO[crop]["seed_cost"]
                need = desired_seeds - seeds.get(crop, 0)
                has_seed = any(m and m[0] == "BUY_SEED" and m[1] == crop for m in action.get("market", []))
                if not has_seed and len(action.get("market", [])) < 10 and cost_s > 0:
                    action["market"].append(["BUY_SEED", crop, min(need, 8 if quads > 1 else 5)])

        # Idle hands parallel planting on empty tiles (nearest to hand, spreads into NE & SW).
        if crop is not None and avail_seeds > 0 and day < 28:
            claimed: set = set()
            new_hands = []
            for h_idx, h_act in enumerate(action.get("hands", [])):
                if h_act == ["PASS"] and avail_seeds > 0:
                    hx, hy = farm["hands"][h_idx]
                    pos = (hx, hy)
                    empties = [pt for pt in unlocked_empty_tiles(farm) if pt not in claimed]
                    if empties:
                        best = min(empties, key=lambda pt: distance(pos, pt))
                        claimed.add(best)
                        if pos == best:
                            new_hands.append(["PLANT", crop])
                            avail_seeds -= 1
                        else:
                            new_hands.append(next_step_toward(pos, best))
                        continue
                new_hands.append(h_act)
            action["hands"] = new_hands

        # Final turn liquidation override.
        observed_step = obs.get("step")
        if observed_step is not None and int(observed_step) >= FINAL_ACTION_STEP:
            market_orders, final_actions = final_turn_liquidation(obs, farm)
            action["farmer"] = final_actions[0]
            action["hands"] = final_actions[1:]
            action["market"] = market_orders[:10]

        return action

    def __call__(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        action = self._base_step(obs)
        return self._apply_shared_overrides(obs, action)


class EvolvedAgent(KaggricultureAgent):
    """
    Evolved champion policy incorporating:
      - 12-animal engine (6 cows, 6 sheep) for consistent cashflow ($1,200+/day)
      - Autonomous milestone land expansion (NE Day 11, SW Day 14)
      - Dynamic market-adaptive crop selection based on real-time prices
      - Protective market filters against dairy baron price crashes
      - Parallel hand planting on empty tiles
      - Strict horizon cutoff on day 28

    Behavior-wise this is now identical to KaggricultureAgent: the previous
    override duplicated __call__ (and had the day-28 cutoff check running
    after the feed-security check instead of before it, plus its own
    hardcoded land-expansion multiplier). Both are fixed by sharing
    KaggricultureAgent's single code path; this subclass now only supplies
    the evolved COMPETITION_CONFIG defaults.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = dict(COMPETITION_CONFIG)
        cfg.update({
            "cash_reserve": 200,
            "land_reserve": 1500,
            "buy_land_cash_multiplier": 1.15,
            "buy_land_enabled": True,
            "expand_day_first": 11,
            "expand_day_second": 14,
            "max_animals": 12,
            "cow_goal": 6,
            "sheep_goal": 6,
            "max_hands": 9,
            "min_cash_for_hand": 50,
            "sell_batch_size": 10,
            "sell_price_ratio": 0.25,
        })
        if config:
            cfg.update(config)
        super().__init__(cfg)

    def _choose_crop_adaptive(self, obs: Dict[str, Any], farm: Dict[str, Any], day: int) -> Optional[str]:
        prices = obs.get("market", {}).get("prices", {})
        wheat_price = prices.get("WHEAT", 25)
        milk_price = prices.get("MILK", 160)
        straw_price = prices.get("STRAWBERRY", 120)
        melon_price = prices.get("MELON", 250)
        carrot_price = prices.get("CARROT", 35)

        # 1. Animal Feed Security: only maintain wheat if livestock is actually profitable
        goals = animal_goals(self.config)
        total_animals = sum(goals.values())
        current_wheat = count_crop(farm, "WHEAT")
        if total_animals > 0 and current_wheat < 4 and (milk_price >= 40 or wheat_price <= 35):
            return "WHEAT"

        # 2. Horizon checks
        if day >= 28:
            return None
        if day >= 26:
            return "WHEAT"
        if day >= 24:
            return "CARROT"

        # 3. Market-Adaptive Selection based on real-time prices
        straw_cnt = count_crop(farm, "STRAWBERRY")
        melon_cnt = count_crop(farm, "MELON")

        # Premium pricing detection
        if straw_price >= 100 and straw_cnt < 28 and day <= 22:
            return "STRAWBERRY"
        if melon_price >= 180 and melon_cnt < 8 and day <= 16:
            return "MELON"
        if carrot_price >= 60 and day <= 25:
            return "CARROT"

        if day <= 16:
            if melon_price >= 160 and melon_cnt < 8:
                if straw_price < 75 or straw_cnt >= 20:
                    return "MELON"
            if straw_price >= 55 and straw_cnt < 24:
                return "STRAWBERRY"
            if melon_cnt < 6 and melon_price >= 140:
                return "MELON"
            return "CARROT"
        elif day <= 23:
            if straw_price >= 65 and straw_cnt < 25 and day <= 20:
                return "STRAWBERRY"
            return "CARROT"
        else:
            return "CARROT"

    def _base_step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Dynamically modulate animal goals based on town shop demand and real-time prices!
        unlocked_shops = set(obs.get("town", {}).get("unlocked_shops", []))
        prices = obs.get("market", {}).get("prices", {})
        wool_p = prices.get("WOOL", 200)
        milk_p = prices.get("MILK", 160)

        # Wool demand: YARN_STORE is the only shop demanding wool.
        has_yarn = "YARN_STORE" in unlocked_shops
        # Milk demand: PIZZA_SHOP or ICE_CREAM_SHOP demand milk.
        has_dairy = ("PIZZA_SHOP" in unlocked_shops) or ("ICE_CREAM_SHOP" in unlocked_shops)

        # If no town shop demands wool and price is depressed, sheep are a trap
        target_sheep = 6 if (has_yarn or wool_p >= 170) else (2 if wool_p >= 120 else 0)
        # Milk is much safer (demanded by multiple shops), but avoid overinvesting if crashed
        target_cows = 8 if (has_dairy or milk_p >= 130) else (4 if milk_p >= 90 else 0)

        self.config["sheep_goal"] = target_sheep
        self.config["cow_goal"] = target_cows
        self.config["max_animals"] = target_sheep + target_cows

        action = super()._base_step(obs)

        # Resilient shed selling: prevent shed congestion from discarding high-value produce
        shed = obs.get("private", {}).get("shed", {})
        shed_total = sum(shed.values())
        if shed_total > 25:
            for item in ["WOOL", "MILK", "FERTILIZER"]:
                qty = shed.get(item, 0)
                p = prices.get(item, 0)
                if qty > 4 and p > 0:
                    has_sell = any(m and m[0] == "SELL" and m[1] == item for m in action.get("market", []))
                    if not has_sell and len(action.get("market", [])) < 10:
                        action["market"].append(["SELL", item, min(qty, 4)])

        return action

    def __call__(self, obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
        player = obs["player"]
        farm = obs["farms"][player]
        money = farm.get("money", 0)
        day = int(obs.get("day", 0))
        hour = int(obs.get("hour", 0))
        quads = len(farm.get("unlocked_quadrants", []))

        action = super().__call__(obs)

        # Protect against crashing livestock market (e.g. dairy opponent)
        prices = obs.get("market", {}).get("prices", {})
        milk_p = prices.get("MILK", 160)
        wool_p = prices.get("WOOL", 200)
        wheat_p = prices.get("WHEAT", 25)
        if milk_p < 30 and wool_p < 30 and wheat_p > milk_p:
            action["market"] = [m for m in action.get("market", []) if not (m and m[0] == "BUY_PRODUCT" and m[1] == "WHEAT")]
        if milk_p < 40:
            action["market"] = [m for m in action.get("market", []) if not (m and m[0] == "BUY_ANIMAL" and m[1] == "COW")]
        if wool_p < 40:
            action["market"] = [m for m in action.get("market", []) if not (m and m[0] == "BUY_ANIMAL" and m[1] == "SHEEP")]

        # Milestone & Saturation Land Expansion
        cost = next_land_cost(farm)
        buy_land_enabled = self.config.get("buy_land_enabled", True)
        first_day = self.config.get("expand_day_first", 11)
        second_day = self.config.get("expand_day_second", 14)

        if buy_land_enabled and cost is not None and day <= 20 and quads < 3:
            empty = count_empty_tiles(farm)
            thresh = max(self.config.get("land_reserve", 1600), cost * 1.20 + self.config.get("cash_reserve", 180))
            should_buy = False
            if quads == 1 and day >= first_day and money >= thresh and empty <= 12:
                should_buy = True
            elif quads == 2 and day >= second_day and (money >= 6000 or (money >= thresh and empty <= 26)):
                should_buy = True

            if should_buy:
                has_land = any(m and m[0] == "BUY_LAND" for m in action.get("market", []))
                if not has_land and len(action.get("market", [])) < 10:
                    action["market"].append(["BUY_LAND"])

        # Scaled Workforce Hiring up to 9 hands
        max_h = self.config.get("max_hands", 9)
        current_hires = int(farm.get("hires_today", 0))
        target_hands = min(max_h, 6 if day < 5 else (8 if day < 9 else (max_h if day < 28 else 6)))
        if current_hires < target_hands and money >= self.config.get("min_cash_for_hand", 50):
            has_hire = any(m and m[0] == "HIRE" for m in action.get("market", []))
            if not has_hire and len(action.get("market", [])) < 10:
                action["market"].append(["HIRE"])

        # Adaptive crop selection
        crop = self._choose_crop_adaptive(obs, farm, day)
        seeds = obs["private"].get("seeds", {})
        avail_seeds = seeds.get(crop, 0) if crop else 0

        # Maintain seed buffer (do not buy seeds after day 28!)
        if crop is not None and day < 28:
            desired_seeds = min(18, max(4, count_empty_tiles(farm))) if quads > 1 else 4
            if seeds.get(crop, 0) < desired_seeds and money >= self.config.get("cash_reserve", 180) + 80:
                cost_s = CROP_INFO[crop]["seed_cost"]
                need = desired_seeds - seeds.get(crop, 0)
                has_seed = any(m and m[0] == "BUY_SEED" and m[1] == crop for m in action.get("market", []))
                if not has_seed and len(action.get("market", [])) < 10 and cost_s > 0:
                    action["market"].append(["BUY_SEED", crop, min(need, 8 if quads > 1 else 5)])

        # Idle hands parallel planting on empty tiles (nearest to hand, spreading into NE & SW)
        if crop is not None and avail_seeds > 0 and day < 28:
            claimed = set()
            new_hands = []
            for h_idx, h_act in enumerate(action.get("hands", [])):
                if h_act == ["PASS"] and avail_seeds > 0:
                    hx, hy = farm["hands"][h_idx]
                    pos = (hx, hy)
                    empties = [pt for pt in unlocked_empty_tiles(farm) if pt not in claimed]
                    if empties:
                        best = min(empties, key=lambda pt: distance(pos, pt))
                        claimed.add(best)
                        if pos == best:
                            new_hands.append(["PLANT", crop])
                            avail_seeds -= 1
                        else:
                            new_hands.append(next_step_toward(pos, best))
                        continue
                new_hands.append(h_act)
            action["hands"] = new_hands

        # Final turn liquidation override
        observed_step = obs.get("step")
        if observed_step is not None and int(observed_step) >= FINAL_ACTION_STEP:
            market_orders, final_actions = final_turn_liquidation(obs, farm)
            action["farmer"] = final_actions[0]
            action["hands"] = final_actions[1:]
            action["market"] = market_orders[:10]

        return action


# ------------------------------------------------------------
# Grandmaster Double Agent Self-Play Champion
# ------------------------------------------------------------

CHAMPION_GENOME: Dict[str, Any] = {
    "base_sheep_goal": 3,
    "base_cow_goal": 3,
    "base_goose_goal": 3,
    "day0_hires": 3,
    "day0_animal": "SHEEP",
    "day0_animal_qty": 3,
    "cash_reserve": 432,
    "max_hands_p1": 3,
    "max_hands_p2": 6,
    "max_hands_p3": 7,
    "max_hands_p4": 12,
    "min_cash_for_hand": 25,
    "expand_day_first": 8,
    "expand_day_second": 12,
    "land_reserve": 542,
    "buy_land_cash_multiplier": 1.09,
    "buy_land_enabled": True,
    "max_wheat_tiles": 2,
    "melon_weight": 2.77,
    "strawberry_weight": 1.97,
    "carrot_weight": 1.3,
    "sell_price_ratio": 0.22,
    "sell_batch_size": 8,
    "drop_threshold": 1,
    "feed_reserve_mult": 1.16
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
        self.config = dict(CHAMPION_GENOME)
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



# ------------------------------------------------------------
# Competition entry point
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
