import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from evoexperiments.double_agent_engine import DoubleAgentPolicy
from experiments.opponents import replay_action_agent

bot = DoubleAgentPolicy()
ht_agent = replay_action_agent("replays/kaggle_replays/113441225.json", 1)
env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 1000}, debug=False)
env.run([lambda obs: bot(obs), ht_agent])

for step in range(0, 720, 24):
    f = env.steps[step]
    obs = f[0]['observation']
    m0 = obs['farms'][0]['money']
    m1 = obs['farms'][1]['money']
    wool = obs['market']['prices']['WOOL']
    fert = obs['market']['prices']['FERTILIZER']
    wheat = obs['market']['prices']['WHEAT']
    print(f"Day {step//24:2d}: P0 Money=${m0:6.0f} | P1 Money=${m1:6.0f} | Wool=${wool:3d} | Fert=${fert:3d} | Wheat=${wheat:3d}")
