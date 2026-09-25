"""Offline policy configuration and explicit evolutionary search space."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "baseline_config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "cash_reserve": 500, "land_reserve": 1000, "max_hands": 3,
    "min_cash_for_hand": 800, "sell_price_ratio": 0.80,
    "sell_min_price": 1, "sell_batch_size": 10,
    "town_demand_weight": 1.0, "town_price_weight": 0.30,
    "town_scarcity_weight": 0.20, "wheat_ratio": 1.0,
    "carrot_ratio": 1.0, "tomato_ratio": 1.0,
    "strawberry_ratio": 1.0, "melon_ratio": 0.80,
    "harvest_buffer_days": 0, "buy_land_enabled": True,
    "buy_land_cash_multiplier": 1.50, "hire_enabled": True,
    "prefer_nearest_action": True, "max_animals": 0, "animal_species": "SHEEP",
    "sheep_goal": 0, "cow_goal": 0,
}

# kind, lower/upper bounds or categorical choices, mutation scale
PARAMETERS: dict[str, dict[str, Any]] = {
    "cash_reserve": {"type": "int", "min": 100, "max": 1500},
    "land_reserve": {"type": "int", "min": 0, "max": 2500},
    "max_hands": {"type": "int", "min": 0, "max": 8},
    "min_cash_for_hand": {"type": "int", "min": 0, "max": 2500},
    "sell_price_ratio": {"type": "float", "min": 0.50, "max": 1.00},
    "sell_batch_size": {"type": "int", "min": 1, "max": 25},
    "buy_land_cash_multiplier": {"type": "float", "min": 1.0, "max": 3.0},
    "town_demand_weight": {"type": "float", "min": 0.0, "max": 3.0},
    "town_price_weight": {"type": "float", "min": 0.0, "max": 1.5},
    "town_scarcity_weight": {"type": "float", "min": 0.0, "max": 1.5},
    **{f"{c}_ratio": {"type": "float", "min": 0.25, "max": 2.0}
       for c in ("wheat", "carrot", "tomato", "strawberry", "melon")},
    "buy_land_enabled": {"type": "bool"},
    "hire_enabled": {"type": "bool"},
    "prefer_nearest_action": {"type": "bool"},
    "max_animals": {"type": "int", "min": 0, "max": 16},
    "animal_species": {"type": "categorical", "choices": ["SHEEP", "COW", "GOOSE"]},
    "sheep_goal": {"type": "int", "min": 0, "max": 16},
    "cow_goal": {"type": "int", "min": 0, "max": 8},
}

TRAIN_SEEDS = tuple(range(1000, 1010))
VALIDATION_SEEDS = tuple(range(2000, 2005))
HOLDOUT_SEEDS = tuple(range(3000, 3005))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load JSON overrides on top of the immutable manual baseline."""
    result = dict(DEFAULT_CONFIG)
    source = Path(path) if path else BASELINE_PATH
    if source.exists():
        with source.open(encoding="utf-8") as stream:
            overrides = json.load(stream)
        if not isinstance(overrides, dict):
            raise ValueError(f"Configuration must be a JSON object: {source}")
        unknown = set(overrides) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        for key, value in overrides.items():
            default = DEFAULT_CONFIG[key]
            if isinstance(default, bool):
                valid = isinstance(value, bool)
            elif isinstance(default, int):
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif isinstance(default, float):
                valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            else:
                valid = isinstance(value, type(default))
            if not valid:
                raise ValueError(f"Invalid type for configuration key {key!r}")
            spec = PARAMETERS.get(key)
            if spec and spec["type"] == "categorical" and value not in spec["choices"]:
                raise ValueError(f"Invalid choice for configuration key {key!r}: {value!r}")
            if spec and spec["type"] in {"float", "int"}:
                if not spec["min"] <= value <= spec["max"]:
                    raise ValueError(f"Configuration value {key!r} must be in "
                                     f"[{spec['min']}, {spec['max']}]")
        result.update(overrides)
    return result
