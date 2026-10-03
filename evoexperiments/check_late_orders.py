import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from evoexperiments.double_agent_policy import DoubleAgent

bot = DoubleAgent()
env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 1000}, debug=False)
env.run([lambda obs: bot(obs), 'starter'])

for step in range(22*24, 30*24):
    act = env.steps[step][0].get('action') or {}
    orders = act.get('market', [])
    non_sell = [o for o in orders if o and o[0] != 'SELL']
    if non_sell:
        obs = env.steps[step][0]['observation']
        m = obs['farms'][0]['money']
        print(f"Step {step} (Day {step//24}, Hr {step%24}): Money=${m} Orders: {non_sell}")
