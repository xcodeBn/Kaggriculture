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
    name_clean = name.strip()
    # Check if it's a file path
    path = Path(name_clean)
    if path.exists() and path.suffix == ".py":
        return (path.name, str(path.resolve()))

    name_lower = name_clean.lower()
    if name_lower in ("submission", "champion", "main", "ours", "main.py", "submission.py"):
        return ("Our Champion (main.py)", str((ROOT / "main.py").resolve()))
    elif name_lower in ("dairy", "dairybaron", "baron"):
        bot = DairyBaronAgent()
        return ("DairyBaron", lambda obs, cfg=None: bot(obs))
    elif name_lower in ("swarm", "swarmplanter"):
        bot = SwarmPlanterAgent()
        return ("SwarmPlanter", lambda obs, cfg=None: bot(obs))
    elif name_lower in ("starter", "starter_bot"):
        return ("Starter Baseline", "starter")
    elif name_lower in ("random", "random_bot"):
        return ("Random Baseline", "random")
    elif name_lower in ("pass", "idle"):
        return ("Pass Agent", "pass")
    else:
        print(f"Unknown agent '{name}', falling back to main.py")
        return ("Our Champion (main.py)", str((ROOT / "main.py").resolve()))

def main():
    parser = argparse.ArgumentParser(
        description="Watch two Kaggriculture agents compete in your browser.",
        epilog="Examples:\n"
               "  python3 watch_game.py main.py starter 1234\n"
               "  python3 watch_game.py --p0 main.py --p1 starter --seed 1234\n"
    )
    # Positional args (optional)
    parser.add_argument("pos_p0", nargs="?", default=None, help="Player 0 (e.g. main.py, starter, dairy, random)")
    parser.add_argument("pos_p1", nargs="?", default=None, help="Player 1 (e.g. starter, dairy, random)")
    parser.add_argument("pos_seed", nargs="?", default=None, type=int, help="Game seed")

    # Flag args
    parser.add_argument("--p0", default=None, help="Player 0 agent")
    parser.add_argument("--p1", default=None, help="Player 1 agent")
    parser.add_argument("--seed", type=int, default=42, help="Game seed (default: 42)")
    parser.add_argument("--steps", type=int, default=720, help="Episode steps (default: 720)")
    parser.add_argument("--no-open", action="store_true", help="Do not auto-open browser")
    parser.add_argument("--out", default="match.html", help="Output HTML filename")
    args = parser.parse_args()

    # Resolve p0, p1, seed from positional or flagged args
    p0_arg = args.p0 or args.pos_p0 or "champion"
    p1_arg = args.p1 or args.pos_p1 or "dairy"
    seed_arg = args.pos_seed if args.pos_seed is not None else args.seed

    p0_name, p0_fn = parse_agent(p0_arg)
    p1_name, p1_fn = parse_agent(p1_arg)
    game_seed = seed_arg

    print(f"\n========================================================")
    print(f" 🌾 Kaggriculture Browser Match Viewer 🌾")
    print(f"========================================================")
    print(f" Player 0 (Blue):  {p0_name}")
    print(f" Player 1 (Orange):{p1_name}")
    print(f" Seed:             {game_seed}")
    print(f" Steps:            {args.steps} (30 in-game days)")
    print(f" Simulating match, please wait...")

    make = get_clean_make()
    env = make("kaggriculture", configuration={"episodeSteps": args.steps, "seed": game_seed}, debug=False)
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
