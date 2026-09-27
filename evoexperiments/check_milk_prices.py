import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make
import evoexperiments.enhanced_agent as enhanced_agent

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 3000}, debug=True)
env.run([enhanced_agent.agent, "starter"])

for step_idx in [100, 200, 300, 400, 500, 600, 700, 719]:
    obs0 = env.steps[step_idx][0].observation
    mkt = obs0["market"]
    shed = obs0["private"]["shed"]
    print(f"Step {step_idx:3d}: MILK Price=${mkt['prices'].get('MILK')} Inv={mkt['inventory'].get('MILK')} | Shed Milk={shed.get('MILK')} | WOOL Price=${mkt['prices'].get('WOOL')} Inv={mkt['inventory'].get('WOOL')} | Shed Wool={shed.get('WOOL')}")
