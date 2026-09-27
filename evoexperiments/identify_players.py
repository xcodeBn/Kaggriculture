"""Identify our player seat and opponent in each Kaggle replay."""
import json
from pathlib import Path

replays_dir = Path("replays/kaggle_replays")
replays = sorted(replays_dir.glob("*.json"))

def identify_players():
    print(f"Checking {len(replays)} replay files...")
    
    seen_episodes = set()
    rows = []
    
    for rpath in replays:
        if rpath.name == "kaggle_game.json":
            continue
            
        with open(rpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        info = data.get("info", {})
        ep_id = info.get("EpisodeId", rpath.stem)
        if ep_id in seen_episodes:
            print(f"Skipping duplicate episode: {ep_id} ({rpath.name})")
            continue
        seen_episodes.add(ep_id)
        
        team_names = info.get("TeamNames", [])
        # Check players in steps
        steps = data.get("steps", [])
        if not steps:
            continue
            
        last_frame = steps[-1]
        p0_reward = last_frame[0].get("reward", 0)
        p1_reward = last_frame[1].get("reward", 0)
        
        # Determine which player is ours.
        # Our team is typically "Hassan Bazzoun-dev" or has 8 cows in recent games.
        our_name = "Hassan Bazzoun-dev"
        our_seat = None
        opp_name = None
        is_self_play = False
        
        if len(team_names) >= 2:
            if team_names[0] == team_names[1]:
                is_self_play = True
                our_seat = 0
                opp_name = team_names[1] + " (Self-play)"
            elif team_names[0] == our_name:
                our_seat = 0
                opp_name = team_names[1]
            elif team_names[1] == our_name:
                our_seat = 1
                opp_name = team_names[0]
            else:
                # Could be earlier submission
                opp_name = f"{team_names[0]} vs {team_names[1]}"
                
        rows.append({
            "file": rpath.name,
            "episode_id": ep_id,
            "our_seat": our_seat,
            "team0": team_names[0] if len(team_names) > 0 else "Unknown",
            "team1": team_names[1] if len(team_names) > 1 else "Unknown",
            "opp_name": opp_name,
            "is_self_play": is_self_play,
            "p0_reward": p0_reward,
            "p1_reward": p1_reward,
        })
        
    print(f"\nUnique episodes found: {len(rows)}")
    print(f"{'Episode':<12} | {'File':<18} | {'Our Seat':<8} | {'Opponent':<25} | {'Self-Play':<9} | {'P0 Reward':<10} | {'P1 Reward':<10}")
    print("-" * 105)
    for r in rows:
        print(f"{r['episode_id']:<12} | {r['file']:<18} | {str(r['our_seat']):<8} | {str(r['opp_name']):<25} | {str(r['is_self_play']):<9} | {r['p0_reward']:<10} | {r['p1_reward']:<10}")

if __name__ == "__main__":
    identify_players()
