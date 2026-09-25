"""Reproducible local opponents."""
from __future__ import annotations

import random
import copy
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from submission import CROP_INFO, CROP_ORDER


def seeded_random_agent(episode_seed: int):
    """Match Kaggle Environments' Kaggriculture random policy deterministically.

    The bundled opponent creates an unseeded ``random.Random()`` per action.
    This copies its action probabilities and choices, but seeds the per-turn
    generator from episode seed, step, and player so paired evaluations repeat.
    """
    def agent(obs: dict[str, Any]) -> dict[str, Any]:
        player = int(obs.get("player", 0))
        step = int(obs.get("step", 0))
        rng = random.Random((episode_seed * 1_000_003) ^ (step * 9_176 + player))
        farms = obs.get("farms", [])
        farm = farms[player] if player < len(farms) else None
        private = obs.get("private", {}) or {}
        if farm is None:
            return {"farmer": ["PASS"], "hands": [], "market": []}

        farmer_ops = ["NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "PASS"]
        market = []
        seeds = private.get("seeds", {})
        affordable = [crop for crop in CROP_ORDER
                      if CROP_INFO[crop]["seed_cost"] <= farm.get("money", 0)]
        if affordable and rng.random() < 0.1:
            market.append(["BUY_SEED", rng.choice(affordable), 1])

        available_seeds = [crop for crop, count in seeds.items() if count > 0]
        if available_seeds and rng.random() < 0.3:
            farmer = ["PLANT", rng.choice(available_seeds)]
        else:
            farmer = [rng.choice(farmer_ops)]
        hands = [[rng.choice(farmer_ops)] for _ in farm.get("hands", [])]
        return {"farmer": farmer, "hands": hands, "market": market}

    return agent


@lru_cache(maxsize=16)
def _load_replay_actions(replay_path: str, player_index: int) -> tuple[dict[str, Any], ...]:
    """Cache only the action tape, not the much larger observation history."""
    with Path(replay_path).open(encoding="utf-8") as stream:
        steps = json.load(stream).get("steps", [])
    actions = []
    for frame in steps:
        if isinstance(frame, list) and 0 <= player_index < len(frame):
            action = frame[player_index].get("action") or {}
            actions.append(copy.deepcopy(action))
        else:
            actions.append({"farmer": ["PASS"], "hands": [], "market": []})
    return tuple(actions)


def replay_action_agent(replay_path: str | Path, player_index: int = 1):
    """Replay one recorded player's actions as a fixed offline stress opponent.

    This is useful for checking a policy against a known game plan. It is not
    adaptive and should not be used as the sole opponent for optimization.
    kaggle-environments replay frames store a player's response to observation
    step ``t`` in frame ``t + 1``; the player 0/1 observations are not aligned
    to the action in the same frame.
    """
    frames = _load_replay_actions(str(Path(replay_path).resolve()), player_index)

    def agent(obs: dict[str, Any]) -> dict[str, Any]:
        frame_index = int(obs.get("step", 0)) + 1
        if 0 <= frame_index < len(frames):
            return copy.deepcopy(frames[frame_index])
        return {"farmer": ["PASS"], "hands": [], "market": []}

    return agent
