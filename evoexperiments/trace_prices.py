import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from evoexperiments.double_agent_policy import DoubleAgent

bot = DoubleAgent()
env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 1000}, debug=False)
env.run([lambda obs: bot(obs), 'starter'])

for step in range(0, 720, 48):
    frame = env.steps[step]
    obs = frame[0]['observation']
    m0 = obs['farms'][0]['money']
    p_wool = obs['market']['prices']['WOOL']
    p_fert = obs['market']['prices']['FERTILIZER']
    p_wheat = obs['market']['prices']['WHEAT']
    inv_wool = obs['market']['inventory']['WOOL']
    inv_fert = obs['market']['inventory']['FERTILIZER']
    print(f"Step {step:3d} (Day {step//24:2d}): Money=${m0:6.0f} | Wool: ${p_wool:3d} (Inv={inv_wool}) | Fert: ${p_fert:3d} (Inv={inv_fert}) | Wheat: ${p_wheat:3d}")

print("Final Money:", env.steps[-1][0]['reward'])
