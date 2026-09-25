"""Summarize a Kaggriculture replay's recorded observations and actions.

The analyzer deliberately reports harvested action counts rather than inferring
produce quantities: replay action traces do not encode the yield of a harvest.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission import SHOP_DEMAND

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
ANIMAL_ACTIONS = {"BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER", "CARE"}
CROP_NAMES = {"WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"}
ONE_TIME_CROPS = {"WHEAT", "CARROT", "MELON"}


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _steps(replay: dict[str, Any]) -> list[list[Any]]:
    steps = replay.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Replay has no non-empty 'steps' array")
    return [step if isinstance(step, list) else [step] for step in steps]


def analyze_replay(replay: dict[str, Any], player: int = 0) -> dict[str, Any]:
    steps = _steps(replay)
    action_counts: Counter[str] = Counter()
    market_counts: Counter[str] = Counter()
    plant_actions: Counter[str] = Counter()
    planted: Counter[str] = Counter()
    harvested_actions: Counter[str] = Counter()
    harvested_confirmed: Counter[str] = Counter()
    seed_buys: Counter[str] = Counter()
    sale_orders: Counter[str] = Counter()
    shop_instances: Counter[str] = Counter()
    demand_exposure: Counter[str] = Counter()
    seen_shops: Counter[str] = Counter()
    last_observation: Any = None
    last_states: list[Any] = []
    turn_count = 0
    unit_action_slots = 0
    previous_tiles: dict[tuple[int, int], tuple[str | None, str | None, int]] = {}
    previous_unit_positions: list[Any] = []
    plant_to_weed_events = 0
    plant_to_empty_events = 0

    for frame in steps:
        if player >= len(frame):
            continue
        state = frame[player]
        last_states = frame
        turn_count += 1
        action_tiles = previous_tiles.copy()
        action_positions = list(previous_unit_positions)
        observation = _get(state, "observation", {})
        last_observation = observation
        replay_farms = _get(observation, "farms", []) or []
        if player < len(replay_farms):
            board = _get(replay_farms[player], "tiles", []) or []
            current_tiles: dict[tuple[int, int], tuple[str | None, str | None, int]] = {}
            for y, row in enumerate(board):
                for x, tile in enumerate(row):
                    kind = _get(tile, "kind", None) if isinstance(tile, dict) else None
                    crop = _get(tile, "crop", None) if isinstance(tile, dict) else None
                    yield_units = int(_get(tile, "yield_units", 0) or 0) if isinstance(tile, dict) else 0
                    current_tiles[(x, y)] = (kind, crop, yield_units)
                    previous_kind, _, _ = previous_tiles.get((x, y), (None, None, 0))
                    if previous_kind == "PLANT" and kind == "WEED":
                        plant_to_weed_events += 1
                    elif previous_kind == "PLANT" and kind != "PLANT":
                        plant_to_empty_events += 1
                    if kind == "PLANT" and previous_kind != "PLANT" and crop:
                        planted[str(crop)] += 1
            previous_tiles = current_tiles
            previous_unit_positions = [
                _get(replay_farms[player], "farmer", None),
                *(_get(replay_farms[player], "hands", []) or []),
            ]
        town = _get(observation, "town", {}) or {}
        shops = tuple(_get(town, "unlocked_shops", []) or [])
        # Shops persist; account only for newly observed instances, including duplicates.
        for shop, count in (Counter(shops) - seen_shops).items():
            shop_instances[shop] += count
            for item, qty in SHOP_DEMAND.get(shop, {}).items():
                demand_exposure[item] += count * qty
        seen_shops = Counter(shops)

        action = _get(state, "action", {}) or {}
        farmer_action = _get(action, "farmer", ["PASS"])
        hand_actions = _get(action, "hands", []) or []
        farm_states = _get(observation, "farms", []) or []
        unit_positions: list[Any] = []
        if player < len(farm_states):
            farm_state = farm_states[player]
            unit_positions = [_get(farm_state, "farmer", None),
                              *(_get(farm_state, "hands", []) or [])]
        unit_action_slots += 1 + len(hand_actions)
        confirmed_harvest_tiles: set[tuple[int, int]] = set()
        for unit_index, op in enumerate([farmer_action, *hand_actions]):
            if not isinstance(op, (list, tuple)) or not op:
                continue
            verb = str(op[0]).upper()
            action_counts[verb] += 1
            if verb in MOVES:
                action_counts["MOVE"] += 1
            if verb == "PLANT" and len(op) > 1:
                plant_actions[str(op[1])] += 1
            if verb == "HARVEST":
                tile = None
                farm_state = farm_states[player] if player < len(farm_states) else {}
                xy = action_positions[unit_index] if unit_index < len(action_positions) else (
                    unit_positions[unit_index] if unit_index < len(unit_positions) else None)
                tile = None
                if xy:
                    x, y = xy
                    prior_kind, prior_crop, prior_yield = action_tiles.get((x, y), (None, None, 0))
                    if prior_kind == "PLANT":
                        tile = {"kind": prior_kind, "crop": prior_crop}
                    else:
                        board = _get(farm_state, "tiles", [])
                        try:
                            tile = board[y][x]
                        except (IndexError, TypeError):
                            pass
                crop = _get(tile, "crop", "UNKNOWN")
                harvested_actions[str(crop)] += 1
                if xy:
                    current_tile = None
                    try:
                        current_tile = _get(farm_state, "tiles", [])[xy[1]][xy[0]]
                    except (IndexError, TypeError):
                        pass
                    current_kind = _get(current_tile, "kind", None) if isinstance(current_tile, dict) else None
                    tile_position = (xy[0], xy[1]) if xy else None
                    if (prior_kind == "PLANT" and str(crop) in ONE_TIME_CROPS
                            and current_kind is None and tile_position not in confirmed_harvest_tiles):
                        harvested_confirmed[str(crop)] += 1
                        confirmed_harvest_tiles.add(tile_position)

        market_orders = _get(action, "market", []) or []
        for order in market_orders:
            if not isinstance(order, (list, tuple)) or not order:
                continue
            verb = str(order[0]).upper()
            market_counts[verb] += 1
            market_counts["ORDERS"] += 1
            if verb == "BUY_SEED" and len(order) > 1:
                seed_buys[str(order[1])] += int(order[2]) if len(order) > 2 else 1
            elif verb == "SELL" and len(order) > 1:
                sale_orders[str(order[1])] += int(order[2]) if len(order) > 2 else 1

    states = last_states
    rewards = [_get(state, "reward", None) for state in states]
    reward = rewards[player] if player < len(rewards) else None
    opponent_reward = rewards[1 - player] if len(rewards) > 1 - player else None
    farms = _get(last_observation, "farms", []) or []
    money = [_get(farm, "money", None) for farm in farms]
    final_money = money[player] if player < len(money) else None
    opponent_money = money[1 - player] if len(money) > 1 - player else None
    outcome = None
    if final_money is not None and opponent_money is not None:
        outcome = "win" if final_money > opponent_money else "loss" if final_money < opponent_money else "tie"
    hands_hired = market_counts["HIRE"]
    animal_action_count = sum(action_counts[verb] for verb in ANIMAL_ACTIONS)
    useful = sum(n for verb, n in action_counts.items()
                 if verb not in {"PASS", "MOVE", *MOVES})
    units = unit_action_slots / max(1, turn_count)
    crop_alignment = {
        crop: {"planted": planted[crop], "demand_exposure": demand_exposure[crop]}
        for crop in sorted(set(planted) | set(CROP_NAMES))
    }
    return {
        "player": player, "turns_observed": turn_count,
        "outcome": {"reward": reward, "opponent_reward": opponent_reward,
                    "money": final_money, "opponent_money": opponent_money, "result": outcome},
        "actions": {"counts": dict(action_counts),
                    "move_percent": 100 * action_counts["MOVE"] / max(1, sum(action_counts.values()) - action_counts["MOVE"]),
                    "pass_percent": 100 * action_counts["PASS"] / max(1, unit_action_slots),
                    "animal_action_count": animal_action_count,
                    "market_order_counts": dict(market_counts)},
        "crops": {"plant_actions_attempted_by_crop": dict(plant_actions),
                  "plant_transitions_observed_by_crop": dict(planted),
                  "harvest_actions_attempted_by_crop": dict(harvested_actions),
                  "harvests_confirmed_by_state_change": dict(harvested_confirmed),
                  "production_units": None, "wasted_production_units": None,
                  "plant_to_weed_events": plant_to_weed_events,
                  "plant_disappearances_without_harvest_action": max(
                      0, plant_to_empty_events - sum(harvested_confirmed.values())),
                  "production_note": "Production quantity and wasted units are not encoded. One-time harvests are counted once when a HARVEST coincides with plant removal; ongoing harvests cannot be confirmed from tile state alone. Plant-to-weed transitions are observable, but their cause cannot be inferred."},
        "economy": {"seed_order_units_requested": dict(seed_buys),
                    "sale_order_units_requested": dict(sale_orders),
                    "sale_revenue": None, "sale_prices": None,
                    "land_purchase_orders": market_counts["BUY_LAND"],
                    "hire_orders": hands_hired,
                    "note": "Replay actions record submitted orders; they do not confirm each order's execution."},
        "town": {"shop_instances_observed": dict(shop_instances),
                 "demand_exposure_by_product": dict(demand_exposure),
                 "crop_planting_vs_demand_exposure": crop_alignment,
                 "exposure_note": "Counts demand entries by shop instance observed; not actual town consumption."},
        "labor": {"hands_hired_actions": hands_hired, "units_estimated": units,
                  "pass_actions": action_counts["PASS"], "useful_actions": useful,
                  "utilization_proxy": useful / max(1, unit_action_slots)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--player", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.replay.open(encoding="utf-8") as stream:
        report = analyze_replay(json.load(stream), args.player)
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
