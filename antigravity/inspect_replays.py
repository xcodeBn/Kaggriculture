import json
from pathlib import Path

replays_dir = Path("replays/kaggle_replays")
replay_files = sorted(replays_dir.glob("*.json"))

print(f"Found {len(replay_files)} files in {replays_dir}")

for p in replay_files:
    if p.name == "kaggle_game.json":
        continue
    with open(p) as f:
        data = json.load(f)
    info = data.get("info", {})
    metadata = data.get("metadata", {})
    steps = data.get("steps", [])
    configuration = data.get("configuration", {})
    agents = data.get("agents", [])
    
    print(f"\n--- File: {p.name} (steps: {len(steps)}) ---")
    print(f"Metadata: {metadata}")
    if isinstance(info, dict):
        print(f"Info keys: {list(info.keys())}")
        if "TeamNames" in info:
            print(f"TeamNames: {info.get('TeamNames')}")
        if "EpisodeId" in info:
            print(f"EpisodeId: {info.get('EpisodeId')}")
    print(f"Agents: {agents}")
    
    if steps:
        for f_idx in [0, 1, 2, 3]:
            if f_idx < len(steps):
                print(f"Frame {f_idx}: len={len(steps[f_idx])}")
                for i, s in enumerate(steps[f_idx]):
                    obs = s.get("observation", {})
                    act = s.get("action")
                    status = s.get("status")
                    reward = s.get("reward")
                    obs_step = obs.get("step") if isinstance(obs, dict) else None
                    player_idx = obs.get("player") if isinstance(obs, dict) else None
                    print(f"  P{i}: status={status}, reward={reward}, act={act}, obs.step={obs_step}, obs.player={player_idx}")
    break
