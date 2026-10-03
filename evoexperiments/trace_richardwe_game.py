import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from evoexperiments.double_agent_engine import DoubleAgentPolicy
from experiments.opponents import replay_action_agent

bot = DoubleAgentPolicy()
rich_agent = replay_action_agent("evoexperiments/115702423.json", 0)

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 2049367962}, debug=False)
env.run([lambda obs: bot(obs), rich_agent])

for step in range(0, 720, 24):
    f = env.steps[step]
    obs = f[0]['observation']
    m0 = obs['farms'][0]['money']
    m1 = obs['farms'][1]['money']
    egg = obs['market']['prices']['EGG']
    wool = obs['market']['prices']['WOOL']
    fert = obs['market']['prices']['FERTILIZER']
    wheat = obs['market']['prices']['WHEAT']
    inv_egg = obs['market']['inventory']['EGG']
    quads0 = obs['farms'][0]['unlocked_quadrants']
    print(f"Day {step//24:2d}: P0 Money=${m0:6.0f} (Quads={len(quads0)}) | P1 Money=${m1:6.0f} | Egg=${egg:2d} (Inv={inv_egg}) | Wool=${wool:3d} | Fert=${fert:3d} | Wheat=${wheat:3d}")
