import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make
import evoexperiments.enhanced_agent as enhanced_agent

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 3000}, debug=True)
env.run([enhanced_agent.agent, "starter"])

final = env.steps[-1]
print(f"Player 0 (Enhanced) Reward: ${final[0].reward:,.0f}, Status: {final[0].status}")
print(f"Player 1 (Starter)  Reward: ${final[1].reward:,.0f}, Status: {final[1].status}")
margin = final[0].reward - final[1].reward
print(f"Margin: ${margin:+,.0f}")

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
    print(f"Step {step_idx:3d} (Day {obs0['day']:2d}): Money=${f0['money']:>8,.0f} | Lands={unlocked} | Animals={len(animals_on_tiles):2d} (Cows={animals_on_tiles.count('COW')}, Sheep={animals_on_tiles.count('SHEEP')}) | EmptyStruct={structures_empty} | Shed={p0['shed']} | Weeds={weeds}")
