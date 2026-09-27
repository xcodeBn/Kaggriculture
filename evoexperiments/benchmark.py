"""Benchmark Competing Agent head-to-head against Optimized Config and Replay Opponents.

Runs controlled, paired evaluations across identical seeds using parallel simulation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kaggle_environments

from config import HOLDOUT_SEEDS, TRAIN_SEEDS, VALIDATION_SEEDS, load_config
from experiments.opponents import replay_action_agent, seeded_random_agent
from evoexperiments.competing_agent import agent as competing_agent
from evoexperiments.parallel_eval import parallel_evaluate_config


def run_paired_game(seed: int, agent0, agent1, steps: int = 720) -> dict:
    """Run a single game between two agents and return rewards."""
    env = kaggle_environments.make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    env.run([agent0, agent1])
    last = env.steps[-1]
    p0_reward = float(last[0].get("reward") or 0)
    p1_reward = float(last[1].get("reward") or 0)
    return {
        "seed": seed,
        "p0_reward": p0_reward,
        "p1_reward": p1_reward,
        "margin": p0_reward - p1_reward,
        "winner": 0 if p0_reward > p1_reward else (1 if p1_reward > p0_reward else "tie")
    }


def benchmark_matchup(
    name: str,
    agent0,
    agent1,
    seeds: tuple[int, ...],
    steps: int = 720
) -> dict:
    """Run head-to-head matches across seeds and compute metrics."""
    results = [run_paired_game(s, agent0, agent1, steps) for s in seeds]
    p0_scores = [r["p0_reward"] for r in results]
    p1_scores = [r["p1_reward"] for r in results]
    margins = [r["margin"] for r in results]
    wins = sum(r["winner"] == 0 for r in results)
    losses = sum(r["winner"] == 1 for r in results)
    ties = sum(r["winner"] == "tie" for r in results)

    mean_margin = sum(margins) / len(margins) if margins else 0
    mean_p0 = sum(p0_scores) / len(p0_scores) if p0_scores else 0
    mean_p1 = sum(p1_scores) / len(p1_scores) if p1_scores else 0

    return {
        "matchup": name,
        "games": len(seeds),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate": (wins / len(seeds)) if seeds else 0.0,
        "mean_agent0_reward": mean_p0,
        "mean_agent1_reward": mean_p1,
        "mean_margin": mean_margin,
        "details": results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002, 1003, 1004])
    parser.add_argument("--steps", type=int, default=720)
    args = parser.parse_args()

    print(f"===========================================================")
    print(f"BENCHMARKING COMPETING AGENT (steps={args.steps}, seeds={args.seeds})")
    print(f"===========================================================\n")

    # Load submission agent with optimized_config.json
    from submission import agent as submission_agent, set_policy_config
    opt_config = load_config(ROOT / "optimized_config.json")
    set_policy_config(opt_config)
    opt_agent = submission_agent

    # Matchup 1: Competing Agent (P0) vs Optimized Config (P1)
    print("Running Matchup: Competing Agent vs Current Optimized Config...")
    t0 = time.time()
    res_opt = benchmark_matchup(
        "Competing Agent vs Optimized Config",
        competing_agent,
        opt_agent,
        tuple(args.seeds),
        args.steps
    )
    dt = time.time() - t0
    print(f"Completed in {dt:.2f}s:")
    print(f"  Record: {res_opt['wins']}W - {res_opt['losses']}L - {res_opt['ties']}T (Win Rate: {res_opt['win_rate']*100:.1f}%)")
    print(f"  Mean Margin: ${res_opt['mean_margin']:+,.0f}")
    print(f"  Competing Agent Mean: ${res_opt['mean_agent0_reward']:,.0f} | Optimized Config Mean: ${res_opt['mean_agent1_reward']:,.0f}\n")

    # Matchup 2: Competing Agent vs GusNicho Replay (Episode 113076092)
    replay_path = ROOT / "replays" / "kaggle_replays" / "113076092.json"
    if replay_path.exists():
        print("Running Matchup: Competing Agent vs GusNicho Recorded Actions...")
        opp_replay = replay_action_agent(str(replay_path), player_index=1)
        res_replay = benchmark_matchup(
            "Competing Agent vs GusNicho Replay",
            competing_agent,
            opp_replay,
            tuple(args.seeds),
            args.steps
        )
        print(f"  Record: {res_replay['wins']}W - {res_replay['losses']}L - {res_replay['ties']}T")
        print(f"  Mean Margin: ${res_replay['mean_margin']:+,.0f}")
        print(f"  Competing Agent: ${res_replay['mean_agent0_reward']:,.0f} | GusNicho: ${res_replay['mean_agent1_reward']:,.0f}\n")

    # Matchup 3: Competing Agent vs Seeded Random
    print("Running Matchup: Competing Agent vs Seeded Random...")
    res_rand = benchmark_matchup(
        "Competing Agent vs Seeded Random",
        competing_agent,
        "random",
        tuple(args.seeds),
        args.steps
    )
    print(f"  Record: {res_rand['wins']}W - {res_rand['losses']}L - {res_rand['ties']}T")
    print(f"  Mean Margin: ${res_rand['mean_margin']:+,.0f}\n")


if __name__ == "__main__":
    main()
