import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from experiments.opponents import replay_action_agent
import evoexperiments.enhanced_agent as enhanced_agent

ht_agent = replay_action_agent("replays/kaggle_replays/113441225.json", 1)

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 3000}, debug=True)
env.run([enhanced_agent.agent, ht_agent])

final = env.steps[-1]
print(f"Player 0 (Enhanced) Reward: ${final[0].reward:,.0f}")
print(f"Player 1 (Harry T.) Reward: ${final[1].reward:,.0f}")
print(f"Margin: ${final[0].reward - final[1].reward:+,.0f}")

for step_idx in [100, 200, 300, 400, 500, 600, 719]:
    obs = env.steps[step_idx][0].observation
    f0 = obs["farms"][0]
    f1 = obs["farms"][1]
    p0 = obs["private"]
    mkt = obs["market"]
    
    animals0 = [t["animal"] for row in f0["tiles"] for t in row if isinstance(t, dict) and t.get("animal")]
    animals1 = [t["animal"] for row in f1["tiles"] for t in row if isinstance(t, dict) and t.get("animal")]
    
    print(f"\n--- Step {step_idx:3d} (Day {obs['day']:2d}) ---")
    print(f"  P0 (Enhanced): Money=${f0['money']:>8,.0f} | Lands={f0['unlocked_quadrants']} | Animals={len(animals0)} {animals0[:3]} | Shed={p0['shed']}")
    print(f"  P1 (Harry T.): Money=${f1['money']:>8,.0f} | Lands={f1['unlocked_quadrants']} | Animals={len(animals1)} {animals1[:3]}")
    print(f"  Market Prices: MILK=${mkt['prices'].get('MILK')} WOOL=${mkt['prices'].get('WOOL')} WHEAT=${mkt['prices'].get('WHEAT')}")
