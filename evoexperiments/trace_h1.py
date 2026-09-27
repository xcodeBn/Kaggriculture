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

def trace_step(obs):
    set_policy_config(mixed_config)
    step = obs.get("step", 0)
    p0 = obs["private"]
    f0 = obs["farms"][obs["player"]]
    hands = f0.get("hands", [])
    act = agent(obs)
    if 114 <= step <= 130:
        h1_pos = hands[1] if len(hands) > 1 else None
        h1_inv = p0["inventories"][2] if len(p0["inventories"]) > 2 else None
        h1_act = act['hands'][1] if len(act['hands']) > 1 else None
        print(f"Step {step}: Shed={p0['shed']['SHEEP']} | H1 pos={h1_pos} inv={h1_inv} act={h1_act}")
    return act

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 3000}, debug=True)
env.run([trace_step, "starter"])
