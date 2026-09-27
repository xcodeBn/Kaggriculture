"""Verify replay alignment and action-observation timing in Kaggle replays and live simulation."""
import json
from pathlib import Path
import kaggle_environments

def inspect_replay_timing(replay_path: Path):
    with open(replay_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps", [])
    print(f"=== Inspecting {replay_path.name} (total frames: {len(steps)}) ===")
    
    # Check frames 0, 1, 2, 3
    for f_idx in range(min(5, len(steps))):
        frame = steps[f_idx]
        print(f"--- Frame {f_idx} ---")
        for p_idx in [0, 1]:
            entry = frame[p_idx]
            obs = entry.get("observation", {})
            act = entry.get("action")
            obs_step = obs.get("step") if isinstance(obs, dict) else None
            obs_day = obs.get("day") if isinstance(obs, dict) else None
            obs_hour = obs.get("hour") if isinstance(obs, dict) else None
            print(f"  P{p_idx}: obs.step={obs_step}, day={obs_day}, hour={obs_hour}, action={act}")

def test_live_simulation_timing():
    print("\n=== Live Simulation Timing Test ===")
    p0_calls = []
    p1_calls = []

    def p0_agent(obs):
        step = obs.get("step")
        act = {"farmer": ["EAST"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]] if step == 0 else []}
        p0_calls.append((step, act))
        return act

    def p1_agent(obs):
        step = obs.get("step")
        act = {"farmer": ["SOUTH"], "hands": [], "market": [["BUY_SEED", "CARROT", 1]] if step == 0 else []}
        p1_calls.append((step, act))
        return act

    env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": 10}, debug=True)
    env.run([p0_agent, p1_agent])

    replay = env.toJSON()
    steps = replay.get("steps", [])
    print(f"Agent calls count: P0={len(p0_calls)}, P1={len(p1_calls)}")
    for f_idx, frame in enumerate(steps[:4]):
        print(f"Frame {f_idx}:")
        for p_idx in [0, 1]:
            entry = frame[p_idx]
            obs = entry.get("observation")
            act = entry.get("action")
            obs_step = obs.get("step") if isinstance(obs, dict) else None
            print(f"  P{p_idx}: obs.step={obs_step}, action={act}")

    print("\nVerifying replay_action_agent alignment:")
    # At step 0 of new game:
    # If replay_action_agent uses frames[step + 1] -> frames[0 + 1] = frames[1]['action']
    # In live simulation, what was frames[1]['action']?
    print(f"Frame 1 P0 action: {steps[1][0].get('action')}")
    print(f"Agent P0 step 0 emitted: {p0_calls[0][1]}")
    assert steps[1][0].get('action') == p0_calls[0][1], "Frame 1 P0 action should match Agent P0 step 0!"
    print(f"Frame 1 P1 action: {steps[1][1].get('action')}")
    print(f"Agent P1 step 0 emitted: {p1_calls[0][1]}")
    assert steps[1][1].get('action') == p1_calls[0][1], "Frame 1 P1 action should match Agent P1 step 0!"
    print("SUCCESS: Both P0 and P1 step t actions are recorded in frame t + 1!")

if __name__ == "__main__":
    replay_file = Path("replays/kaggle_replays/113076092.json")
    if replay_file.exists():
        inspect_replay_timing(replay_file)
    test_live_simulation_timing()
