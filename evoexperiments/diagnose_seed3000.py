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

def test_agent(obs):
    set_policy_config(mixed_config)
    return agent(obs)

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 3000}, debug=True)
env.run([test_agent, "starter"])

final = env.steps[-1]
print(f"Player 0 Reward: {final[0].reward}, Status: {final[0].status}")
print(f"Player 1 Reward: {final[1].reward}, Status: {final[1].status}")

# Check key steps
for step_idx in [50, 100, 200, 300, 400, 500, 600, 719]:
    obs0 = env.steps[step_idx][0].observation
    f0 = obs0["farms"][0]
    p0 = obs0["private"]
    animals_on_tiles = [
        tile["animal"] for row in f0["tiles"] for tile in row
        if isinstance(tile, dict) and tile.get("animal")
    ]
    structures_empty = sum(
        1 for row in f0["tiles"] for tile in row
        if isinstance(tile, dict) and tile.get("kind") in ("PASTURE", "COOP") and not tile.get("animal")
    )
    weeds = sum(
        1 for row in f0["tiles"] for tile in row
        if isinstance(tile, dict) and tile.get("kind") == "WEED"
    )
    unlocked = f0.get("unlocked_quadrants", [])
    print(f"Step {step_idx:3d} (Day {obs0['day']}, Hr {obs0['hour']}): Money=${f0['money']:,} | Lands={unlocked} | Animals={len(animals_on_tiles)} {animals_on_tiles[:4]} | EmptyStruct={structures_empty} | Shed={p0['shed']} | Weeds={weeds}")
