import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from evoexperiments.double_agent_engine import DoubleAgentPolicy
from collections import Counter
import time

bot = DoubleAgentPolicy()
t0 = time.time()
env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 1000}, debug=False)
env.run([lambda obs: bot(obs), 'starter'])

p0 = env.steps[-1][0]
p1 = env.steps[-1][1]
r0 = float(p0['reward'] or 0)
r1 = float(p1['reward'] or 0)
dt = time.time() - t0

sales = Counter()
for s in env.steps[1:]:
    act = s[0].get('action') or {}
    for o in act.get('market', []):
        if o and o[0] == 'SELL':
            sales[o[1]] += o[2] if len(o)>2 else 1

print(f"Elapsed: {dt:.2f}s")
print(f"P0 Reward: ${r0:,.0f} | P1 Reward: ${r1:,.0f} | Margin: ${r0-r1:+,.0f}")
print("Sales:", sales.most_common(8))
