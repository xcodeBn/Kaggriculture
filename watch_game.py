#!/usr/bin/env python3
"""Run two Kaggriculture agents against each other and watch the replay in your browser.

Usage:
    python3 watch_game.py                          # Our champion vs DairyBaron
    python3 watch_game.py --p1 starter             # Our champion vs Starter baseline
    python3 watch_game.py --p1 swarm               # Our champion vs SwarmPlanter
    python3 watch_game.py --p0 dairy --p1 swarm    # DairyBaron vs SwarmPlanter
    python3 watch_game.py --seed 1234              # Custom seed
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evoexperiments.env_loader import get_clean_make
import submission
from evoexperiments.league_agents import DairyBaronAgent, SwarmPlanterAgent

def parse_agent(name: str):
    name = name.lower().strip()
    if name in ("submission", "champion", "main", "ours"):
        return ("Our Champion (submission.py)", submission.agent)
    elif name in ("dairy", "dairybaron", "baron"):
        bot = DairyBaronAgent()
        return ("DairyBaron", lambda obs, cfg=None: bot(obs))
    elif name in ("swarm", "swarmplanter"):
        bot = SwarmPlanterAgent()
        return ("SwarmPlanter", lambda obs, cfg=None: bot(obs))
    elif name in ("starter", "starter_bot"):
        return ("Starter Baseline", "starter")
    elif name in ("random", "random_bot"):
        return ("Random Baseline", "random")
    elif name in ("pass", "idle"):
        return ("Pass Agent", "pass")
    else:
        print(f"Unknown agent '{name}', falling back to submission.py")
        return ("Our Champion (submission.py)", submission.agent)

def main():
    parser = argparse.ArgumentParser(description="Watch two Kaggriculture agents compete in your browser.")
    parser.add_argument("--p0", default="champion", help="Player 0 agent: champion, dairy, swarm, starter, random")
    parser.add_argument("--p1", default="dairy", help="Player 1 agent: champion, dairy, swarm, starter, random")
    parser.add_argument("--seed", type=int, default=42, help="Game seed (default: 42)")
    parser.add_argument("--steps", type=int, default=720, help="Episode steps (default: 720)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    parser.add_argument("--out", default="match.html", help="Output HTML filename")
    args = parser.parse_args()

    p0_name, p0_fn = parse_agent(args.p0)
    p1_name, p1_fn = parse_agent(args.p1)

    print(f"\n========================================================")
    print(f" 🌾 Kaggriculture Browser Match Viewer 🌾")
    print(f"========================================================")
    print(f" Player 0 (Blue):  {p0_name}")
    print(f" Player 1 (Orange):{p1_name}")
    print(f" Seed:             {args.seed}")
    print(f" Steps:            {args.steps} (30 in-game days)")
    print(f" Simulating match, please wait...")

    make = get_clean_make()
    env = make("kaggriculture", configuration={"episodeSteps": args.steps, "seed": args.seed}, debug=False)
    env.run([p0_fn, p1_fn])

    final_0 = env.steps[-1][0]
    final_1 = env.steps[-1][1]
    reward_0 = float(final_0.reward or 0)
    reward_1 = float(final_1.reward or 0)

    print(f"\n--- Match Finished ---")
    print(f" Player 0 ({p0_name:24s}): ${reward_0:,.0f}")
    print(f" Player 1 ({p1_name:24s}): ${reward_1:,.0f}")
    if reward_0 > reward_1:
        print(f" 🏆 Winner: Player 0 (+${reward_0 - reward_1:,.0f} margin)")
    elif reward_1 > reward_0:
        print(f" 🏆 Winner: Player 1 (+${reward_1 - reward_0:,.0f} margin)")
    else:
        print(f" 🤝 Tie game!")

    out_file = Path(args.out).resolve()
    print(f"\nGenerating interactive browser visualizer...")
    html_content = env.render(mode="html", width=1280, height=820)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f" Saved replay visualizer to: {out_file}")

    if not args.no_open:
        print(f" Opening in your default web browser...")
        if sys.platform == "darwin":
            subprocess.run(["open", str(out_file)])
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(out_file)])
        elif sys.platform == "win32":
            os.startfile(str(out_file))

    print(f"\nYou can now scrub through the game, zoom in on farms, and watch your agents play!\n")

if __name__ == "__main__":
    main()
