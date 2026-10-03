import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from evoexperiments.double_agent_engine import DoubleAgentPolicy
from experiments.opponents import replay_action_agent
import time

bot = DoubleAgentPolicy()
t0 = time.time()
gus_agent = replay_action_agent("replays/kaggle_replays/113076092.json", 1)
env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 1000}, debug=False)
env.run([lambda obs: bot(obs), gus_agent])

p0 = env.steps[-1][0]
p1 = env.steps[-1][1]
r0 = float(p0['reward'] or 0)
r1 = float(p1['reward'] or 0)
dt = time.time() - t0

print(f"Elapsed: {dt:.2f}s")
print(f"DoubleAgent P0 Reward: ${r0:,.0f} | GusNicho P1 Reward: ${r1:,.0f} | Margin: ${r0-r1:+,.0f}")
