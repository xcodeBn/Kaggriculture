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
            min(10, max(4, count_empty_tiles(farm))) if quads > 1 else 4
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
                5,
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

            # Dig weed on unlocked tiles
            if weeds:
                nearest_w = min(weeds, key=lambda w: distance(hand_position, w))
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
            desired_seeds = min(10, max(4, count_empty_tiles(farm))) if quads > 1 else 4
            if seeds.get(crop, 0) < desired_seeds and money >= self.config.get("cash_reserve", 180) + 80:
                cost_s = CROP_INFO[crop]["seed_cost"]
                need = desired_seeds - seeds.get(crop, 0)
                has_seed = any(m and m[0] == "BUY_SEED" and m[1] == crop for m in action.get("market", []))
                if not has_seed and len(action.get("market", [])) < 10 and cost_s > 0:
                    action["market"].append(["BUY_SEED", crop, min(need, 5)])

        # Idle hands parallel planting on empty tiles.
        if crop is not None and avail_seeds > 0 and day < 28:
            claimed: set = set()
            new_hands = []
            for h_idx, h_act in enumerate(action.get("hands", [])):
                if h_act == ["PASS"] and avail_seeds > 0:
                    hx, hy = farm["hands"][h_idx]
                    pos = (hx, hy)
                    empties = [pt for pt in unlocked_empty_tiles(farm) if pt not in claimed]
                    if empties:
                        best = min(empties, key=lambda pt: (abs(pt[0] - 4.5) + abs(pt[1] - 4.5), distance(pos, pt)))
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


# ------------------------------------------------------------
# Competition entry point
# ------------------------------------------------------------

_agent = EvolvedAgent(COMPETITION_CONFIG)


def agent(obs: Dict[str, Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    return _agent(obs, *args, **kwargs)


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