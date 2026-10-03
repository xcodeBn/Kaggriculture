import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from evoexperiments.double_agent_engine import DoubleAgentPolicy
from experiments.opponents import replay_action_agent
from collections import Counter
import time

bot = DoubleAgentPolicy()
rich_agent = replay_action_agent("evoexperiments/115702423.json", 0)

t0 = time.time()
env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 2049367962}, debug=False)
env.run([lambda obs: bot(obs), rich_agent])

r0 = float(env.steps[-1][0].reward or 0)
r1 = float(env.steps[-1][1].reward or 0)
dt = time.time() - t0

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

print(f"Elapsed: {dt:.2f}s")
print(f"Seed 2049367962 Matchup:")
print(f"  New Central-Ring DoubleAgent (P0): ${r0:,.0f}")
print(f"  Richardwe Replay (P1):             ${r1:,.0f}")
print(f"  Margin:                            ${r0 - r1:+,.0f}")
print(f"Sales: {sales.most_common(8)}")
print(f"Unit Actions: {unit_acts.most_common(10)}")
