"""Measure observed market price moves and same-turn replay sale requests.

Replay actions submitted against observation step ``t`` are stored in frame
``t + 1``. Prices and inventory are taken from observation ``t`` and ``t + 1``;
sale requests are read from that aligned action frame. Orders are requests, not
proof that the full quantity executed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = ROOT / "replays" / "kaggle_replays"
DEFAULT_OUTPUT = ROOT / "analysis" / "market_shock_findings.json"
BASE_PRICES = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
    "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200,
    "FERTILIZER": 100,
}


def _market(frame: list[Any]) -> dict[str, Any]:
    if not frame:
        return {}
    state = frame[0] if isinstance(frame[0], dict) else {}
    observation = state.get("observation") or {}
    return observation.get("market") or {}


def _sell_requests(frame: list[Any], names: list[str]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for player, state in enumerate(frame):
        if not isinstance(state, dict):
            continue
        orders = (state.get("action") or {}).get("market") or []
        quantities: Counter[str] = Counter()
        for order in orders:
            if (isinstance(order, (list, tuple)) and len(order) >= 3
                    and order[0] == "SELL" and order[1] in BASE_PRICES):
                try:
                    quantities[str(order[1])] += max(0, int(order[2]))
                except (TypeError, ValueError):
                    continue
        name = names[player] if player < len(names) else f"player_{player}"
        result[name] = dict(quantities)
    return result


def analyze_market_replay(replay: dict[str, Any], source: str = "") -> dict[str, Any]:
    steps = replay.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError("Replay needs at least two frames")
    names = [entry.get("Name", f"player_{i}")
             for i, entry in enumerate((replay.get("info", {}) or {}).get("Agents", []))]
    per_product = {
        item: {"price_observations": 0, "min_price": None, "floor_frames": 0,
              "below_half_base_frames": 0, "price_decreases": 0,
              "decreases_with_sell_requests": 0, "sell_turns_requested": 0,
              "sell_units_requested": 0, "requested_units_on_decrease_turns": 0}
        for item in BASE_PRICES
    }
    events = []
    for step in range(min(719, len(steps) - 1)):
        before = _market(steps[step])
        after = _market(steps[step + 1])
        bp = before.get("prices") or {}
        ap = after.get("prices") or {}
        bi = before.get("inventory") or {}
        ai = after.get("inventory") or {}
        # The action in the next frame was submitted against this frame's
        # observation. This offset was verified by same-seed full replay.
        requests = _sell_requests(steps[step + 1], names)
        for item, base in BASE_PRICES.items():
            if item not in bp or item not in ap:
                continue
            stats = per_product[item]
            price = int(bp[item])
            next_price = int(ap[item])
            stats["price_observations"] += 1
            stats["min_price"] = price if stats["min_price"] is None else min(stats["min_price"], price)
            stats["floor_frames"] += price <= 1
            stats["below_half_base_frames"] += price < base * 0.5
            requested = {name: products.get(item, 0) for name, products in requests.items()
                         if products.get(item, 0) > 0}
            total_requested = sum(requested.values())
            stats["sell_turns_requested"] += total_requested > 0
            stats["sell_units_requested"] += total_requested
            delta = next_price - price
            if delta < 0:
                stats["price_decreases"] += 1
                if total_requested:
                    stats["decreases_with_sell_requests"] += 1
                    stats["requested_units_on_decrease_turns"] += total_requested
                    if -delta >= 20:
                        events.append({
                            "episode_id": (replay.get("info", {}) or {}).get("EpisodeId", replay.get("id")),
                            "source": source, "step": step, "product": item,
                            "price_before": price, "price_after": next_price,
                            "inventory_before": bi.get(item), "inventory_after": ai.get(item),
                            "sell_units_requested_by_player": requested,
                        })
    events.sort(key=lambda event: (event["price_before"] - event["price_after"]), reverse=True)
    return {
        "episode_id": (replay.get("info", {}) or {}).get("EpisodeId", replay.get("id")),
        "source": source,
        "players": names,
        "frames": len(steps),
        "products": per_product,
        "largest_same_turn_drops": events[:20],
    }


def analyze_paths(paths: Iterable[Path]) -> dict[str, Any]:
    episodes = []
    seen = set()
    for path in sorted(paths):
        with path.open(encoding="utf-8") as stream:
            replay = json.load(stream)
        episode_id = (replay.get("info", {}) or {}).get("EpisodeId", replay.get("id", path.name))
        if episode_id in seen:
            continue
        seen.add(episode_id)
        episodes.append(analyze_market_replay(replay, str(path)))

    aggregate = {}
    for item in BASE_PRICES:
        records = [episode["products"][item] for episode in episodes]
        sum_keys = ("price_observations", "floor_frames", "below_half_base_frames",
                    "price_decreases", "decreases_with_sell_requests", "sell_turns_requested",
                    "sell_units_requested", "requested_units_on_decrease_turns")
        aggregate[item] = {key: sum(int(record[key]) for record in records) for key in sum_keys}
        aggregate[item]["min_price"] = min((record["min_price"] for record in records
                                             if record["min_price"] is not None), default=None)
    all_events = [event for episode in episodes for event in episode["largest_same_turn_drops"]]
    all_events.sort(key=lambda event: event["price_before"] - event["price_after"], reverse=True)
    return {"episodes": episodes, "episode_count": len(episodes),
            "aggregate": aggregate, "largest_same_turn_drops": all_events[:30]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="*", type=Path,
                        help="replay JSON files (defaults to every JSON file in replays/kaggle_replays)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = args.replays or list(DEFAULT_REPLAY_DIR.glob("*.json"))
    report = analyze_paths(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"episodes": report["episode_count"], "output": str(args.output),
                      "aggregate": report["aggregate"],
                      "largest_same_turn_drops": report["largest_same_turn_drops"][:8]}, indent=2))


if __name__ == "__main__":
    main()
