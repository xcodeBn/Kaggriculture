import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make

def test_hand_actions(obs):
    step = obs.get("step", 0)
    if step == 0:
        return {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]]}
    if step == 1:
        return {"farmer": ["PASS"], "hands": [["BUILD_PASTURE"]], "market": []}
    return {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}

env = make("kaggriculture", configuration={"episodeSteps": 10}, debug=True)
env.run([test_hand_actions, "pass"])
for log in env.steps[2][0].get("logs", []):
    print("LOG:", log)
