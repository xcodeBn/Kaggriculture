import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from evoexperiments.double_agent_policy import DoubleAgent
from collections import Counter
import time

t0 = time.time()
bot = DoubleAgent()
env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 1000}, debug=False)
env.run([lambda obs: bot(obs), 'starter'])

p0 = env.steps[-1][0]
p1 = env.steps[-1][1]

sales = Counter()
unit_acts = Counter()
for s in env.steps[1:]:
    act = s[0].get('action') or {}
    farmer = act.get('farmer', [])
    if farmer: unit_acts[farmer[0]] += 1
    for h in act.get('hands', []):
        if h: unit_acts[h[0]] += 1
    for o in act.get('market', []):
        if o and o[0] == 'SELL':
            sales[o[1]] += o[2] if len(o)>2 else 1

dt = time.time() - t0
r0 = float(p0['reward'] or 0)
r1 = float(p1['reward'] or 0)
print(f"Elapsed: {dt:.2f}s")
print(f"DoubleAgent P0 Reward: ${r0:,.0f}")
print(f"Starter P1 Reward:     ${r1:,.0f}")
print(f"Net Margin:            ${r0 - r1:+,.0f}")
print("Sales breakdown:", sales.most_common(8))
print("Unit actions breakdown:", unit_acts.most_common(12))
