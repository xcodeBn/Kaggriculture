import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from evoexperiments.double_agent_policy import DoubleAgent
from collections import Counter
import time

bot = DoubleAgent({
    "sell_price_ratio": 0.65,
    "sell_batch_size": 8,
})

t0 = time.time()
env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': 1000}, debug=False)
env.run([lambda obs: bot(obs), 'starter'])

dt = time.time() - t0
r0 = float(env.steps[-1][0]['reward'] or 0)
r1 = float(env.steps[-1][1]['reward'] or 0)
print(f"Elapsed: {dt:.2f}s")
print(f"DoubleAgent P0 Reward: ${r0:,.0f}")
print(f"Starter P1 Reward:     ${r1:,.0f}")
print(f"Net Margin:            ${r0 - r1:+,.0f}")

for step in range(0, 720, 72):
    frame = env.steps[step]
    obs = frame[0]['observation']
    m0 = obs['farms'][0]['money']
    p_wool = obs['market']['prices']['WOOL']
    p_fert = obs['market']['prices']['FERTILIZER']
    p_wheat = obs['market']['prices']['WHEAT']
    inv_wool = obs['market']['inventory']['WOOL']
    print(f"Step {step:3d} (Day {step//24:2d}): Money=${m0:6.0f} | Wool: ${p_wool:3d} (Inv={inv_wool}) | Fert: ${p_fert:3d} | Wheat: ${p_wheat:3d}")
