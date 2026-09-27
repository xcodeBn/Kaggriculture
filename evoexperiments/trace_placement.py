import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from config import load_config
from submission import COMPETITION_CONFIG, set_policy_config, agent

mixed_config = load_config(ROOT / "optimized_config.json")
mixed_config.update({
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
    "sell_price_ratio": 0.50,
})

def trace_agent(obs):
    set_policy_config(mixed_config)
    action = agent(obs)
    step = obs.get("step", 0)
    p0 = obs["private"]
    f0 = obs["farms"][obs["player"]]
    hands = f0.get("hands", [])
    if 95 <= step <= 115:
        print(f"\n--- STEP {step} ---")
        print(f"Farmer at {f0['farmer']}, action: {action.get('farmer')}")
        print(f"Shed: {p0['shed']}")
        print(f"Inventories: {p0['inventories']}")
        print(f"Hands at {hands}, actions: {action.get('hands')}")
        print(f"Market: {action.get('market')}")
    return action

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 3000}, debug=True)
env.run([trace_agent, "starter"])
