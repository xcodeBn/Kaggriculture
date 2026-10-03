"""Co-Evolutionary League Training Harness with Progressive Profit Ratchet.

Features:
1. Multi-Agent Co-Evolution: Pits 'SwarmPlanter' vs 'DairyBaron' in direct head-to-head matches.
2. Shared Market Stress Testing: Forces agents to adapt to dynamic price drops caused by the opponent.
3. Benchmark Validation Pool: Tests candidates against top Kaggle replays:
   - Harry Taylor ($99,824) [113441225.json:1]
   - Wataru420 ($86,082) [113724390.json:1]
   - Starter baseline
4. Progressive Profit Ratchet:
   - Fitness prioritizes absolute profit: Fitness = Player Money + 0.5 * Margin.
   - Automatically raises the profit target floor ($70k -> $75k -> $85k -> $95k -> $100k+).
   - Penalizes sub-floor candidates to constantly force evolution toward higher wealth.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evoexperiments.league_agents import (
    SWARM_PLANTER_BASE, DAIRY_BARON_BASE,
    SwarmPlanterAgent, DairyBaronAgent
)

CHECKPOINT_DIR = ROOT / "evoexperiments" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RATCHET_MILESTONES = [70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000]

LEAGUE_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "cash_reserve": {"type": "int", "min": 50, "max": 600},
    "land_reserve": {"type": "int", "min": 500, "max": 4000},
    "expand_day_first": {"type": "int", "min": 7, "max": 14},
    "expand_day_second": {"type": "int", "min": 11, "max": 18},
    "max_hands": {"type": "int", "min": 8, "max": 10},
    "min_cash_for_hand": {"type": "int", "min": 25, "max": 300},
    "cow_goal": {"type": "int", "min": 2, "max": 8},
    "sheep_goal": {"type": "int", "min": 2, "max": 8},
    "strawberry_ratio": {"type": "float", "min": 0.5, "max": 2.5},
    "melon_ratio": {"type": "float", "min": 0.2, "max": 2.0},
    "sell_price_ratio": {"type": "float", "min": 0.15, "max": 0.60},
    "sell_batch_size": {"type": "int", "min": 4, "max": 15},
}


def mutate_config(config: Dict[str, Any], rng: random.Random, rate: float = 0.25) -> Dict[str, Any]:
    child = dict(config)
    for key, spec in LEAGUE_PARAMETERS.items():
        if key not in child or rng.random() >= rate:
            continue
        kind = spec["type"]
        if kind == "int":
            span = max(1, (spec["max"] - spec["min"]) // 5)
            child[key] = max(spec["min"], min(spec["max"], int(child[key]) + rng.randint(-span, span)))
        elif kind == "float":
            span = (spec["max"] - spec["min"]) * 0.15
            child[key] = round(max(spec["min"], min(spec["max"], float(child[key]) + rng.uniform(-span, span))), 2)
    return child


def run_single_league_game(args: Tuple[str, Dict[str, Any], str, Any, int, int]) -> Dict[str, Any]:
    agent_type, agent_cfg, opp_type, opp_data, seed, steps = args
    from evoexperiments.env_loader import get_clean_make
    make = get_clean_make()

    # Instantiate primary agent
    if agent_type == "SwarmPlanter":
        bot = SwarmPlanterAgent(agent_cfg)
    else:
        bot = DairyBaronAgent(agent_cfg)

    # Instantiate opponent
    if opp_type == "SwarmPlanter":
        opp_bot = SwarmPlanterAgent(opp_data)
    elif opp_type == "DairyBaron":
        opp_bot = DairyBaronAgent(opp_data)
    elif opp_type == "starter":
        opp_bot = "starter"
    elif opp_type == "random":
        from experiments.opponents import seeded_random_agent
        opp_bot = seeded_random_agent(seed)
    elif opp_type == "replay":
        from experiments.opponents import replay_action_agent
        replay_path, player_idx = opp_data
        opp_bot = replay_action_agent(replay_path, player_idx)
    else:
        opp_bot = "starter"

    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    env.run([lambda obs: bot(obs), opp_bot if isinstance(opp_bot, str) else (lambda obs: opp_bot(obs))])
    
    final = env.steps[-1]
    p0_reward = float(final[0].reward or 0)
    p1_reward = float(final[1].reward or 0)
    margin = p0_reward - p1_reward
    return {
        "seed": seed,
        "agent_money": p0_reward,
        "opp_money": p1_reward,
        "margin": margin,
        "win": 1 if margin > 0 else 0
    }


def evaluate_candidate_suite(
    agent_type: str,
    candidate_cfg: Dict[str, Any],
    peer_type: str,
    peer_cfg: Dict[str, Any],
    seeds: Tuple[int, ...],
    steps: int = 720
) -> Dict[str, Any]:
    tasks = []
    
    # 1. Head-to-Head Self-Play match
    for s in seeds[:2]:
        tasks.append((agent_type, candidate_cfg, peer_type, peer_cfg, s, steps))

    # 2. Match vs Harry Taylor ($99k)
    ht_path = str(ROOT / "replays" / "kaggle_replays" / "113441225.json")
    if Path(ht_path).exists():
        tasks.append((agent_type, candidate_cfg, "replay", (ht_path, 1), seeds[0], steps))

    # 3. Match vs Wataru420 ($86k)
    wt_path = str(ROOT / "replays" / "kaggle_replays" / "113724390.json")
    if Path(wt_path).exists():
        tasks.append((agent_type, candidate_cfg, "replay", (wt_path, 1), seeds[0], steps))

    # 4. Match vs Starter
    tasks.append((agent_type, candidate_cfg, "starter", None, seeds[0], steps))

    workers = max(1, min(os.cpu_count() or 1, len(tasks)))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_single_league_game, tasks))
    else:
        results = [run_single_league_game(t) for t in tasks]

    avg_money = sum(r["agent_money"] for r in results) / len(results)
    avg_margin = sum(r["margin"] for r in results) / len(results)
    wins = sum(r["win"] for r in results)
    
    return {
        "avg_money": avg_money,
        "avg_margin": avg_margin,
        "wins": wins,
        "games": len(results),
        "results": results
    }


def run_coevolution_league(
    generations: int = 40,
    population_size: int = 4,
    steps: int = 720,
    seed: int = 42
):
    rng = random.Random(seed)
    train_seeds = (1000, 1001, 1002)
    val_seeds = (2000, 2001, 2002)

    ratchet_floor = 65000.0

    champ_swarm = dict(SWARM_PLANTER_BASE)
    champ_dairy = dict(DAIRY_BARON_BASE)

    print("\n=======================================================")
    print(f"STARTING CO-EVOLUTIONARY LEAGUE TRAINING")
    print(f"Generations: {generations} | Population: {population_size} per archetype")
    print(f"Archetypes: SwarmPlanter vs DairyBaron")
    print(f"Initial Profit Ratchet Floor: ${ratchet_floor:,.0f}")
    print("=======================================================\n")

    swarm_pop = [dict(champ_swarm)] + [mutate_config(champ_swarm, rng, 0.35) for _ in range(population_size - 1)]
    dairy_pop = [dict(champ_dairy)] + [mutate_config(champ_dairy, rng, 0.35) for _ in range(population_size - 1)]

    best_overall_money = 0.0

    for gen in range(1, generations + 1):
        t0 = time.time()
        print(f"\n--- Generation {gen}/{generations} [Ratchet Floor: ${ratchet_floor:,.0f}] ---")

        # Evaluate Swarm candidates
        swarm_scores = []
        for s_cfg in swarm_pop:
            eval_res = evaluate_candidate_suite("SwarmPlanter", s_cfg, "DairyBaron", champ_dairy, train_seeds, steps)
            # Ratchet Fitness
            money = eval_res["avg_money"]
            margin = eval_res["avg_margin"]
            ratchet_penalty = max(0.0, ratchet_floor - money) * 2.5
            fitness = money + 0.5 * margin - ratchet_penalty
            swarm_scores.append((fitness, money, margin, s_cfg))

        swarm_scores.sort(key=lambda x: -x[0])
        best_swarm_fit, best_swarm_money, best_swarm_margin, champ_swarm = swarm_scores[0]
        print(f"  SwarmPlanter Top: Money=${best_swarm_money:+,.0f} | Margin=${best_swarm_margin:+,.0f} | Fitness={best_swarm_fit:,.0f}")

        # Evaluate Dairy candidates
        dairy_scores = []
        for d_cfg in dairy_pop:
            eval_res = evaluate_candidate_suite("DairyBaron", d_cfg, "SwarmPlanter", champ_swarm, train_seeds, steps)
            money = eval_res["avg_money"]
            margin = eval_res["avg_margin"]
            ratchet_penalty = max(0.0, ratchet_floor - money) * 2.5
            fitness = money + 0.5 * margin - ratchet_penalty
            dairy_scores.append((fitness, money, margin, d_cfg))

        dairy_scores.sort(key=lambda x: -x[0])
        best_dairy_fit, best_dairy_money, best_dairy_margin, champ_dairy = dairy_scores[0]
        print(f"  DairyBaron   Top: Money=${best_dairy_money:+,.0f} | Margin=${best_dairy_margin:+,.0f} | Fitness={best_dairy_fit:,.0f}")

        # Check for Ratchet Upward Advancement
        gen_top_money = max(best_swarm_money, best_dairy_money)
        if gen_top_money > best_overall_money:
            best_overall_money = gen_top_money

        for milestone in RATCHET_MILESTONES:
            if best_overall_money >= milestone and ratchet_floor < milestone:
                ratchet_floor = float(milestone)
                print(f"\n  🎯 [PROFIT RATCHET ACTIVATED!] New Floor: ${ratchet_floor:,.0f} achieved! (Peak Money: ${best_overall_money:,.0f})\n")

        # Save Checkpoints
        champion_data = {
            "generation": gen,
            "ratchet_floor": ratchet_floor,
            "best_overall_money": best_overall_money,
            "swarm_planter": champ_swarm,
            "dairy_baron": champ_dairy,
        }
        with open(CHECKPOINT_DIR / "best_league_champion.json", "w") as f:
            json.dump(champion_data, f, indent=2)

        # Breed next generation
        new_swarm_pop = [dict(champ_swarm)]
        while len(new_swarm_pop) < population_size:
            parent = rng.choice(swarm_scores[:2])[3]
            new_swarm_pop.append(mutate_config(parent, rng, 0.30))
        swarm_pop = new_swarm_pop

        new_dairy_pop = [dict(champ_dairy)]
        while len(new_dairy_pop) < population_size:
            parent = rng.choice(dairy_scores[:2])[3]
            new_dairy_pop.append(mutate_config(parent, rng, 0.30))
        dairy_pop = new_dairy_pop

        dt = time.time() - t0
        print(f"  Gen {gen} completed in {dt:.1f}s")

    print("\n=======================================================")
    print(f"CO-EVOLUTION LEAGUE COMPLETE!")
    print(f"Peak Wealth Achieved: ${best_overall_money:,.0f}")
    print(f"Final Ratchet Floor:  ${ratchet_floor:,.0f}")
    print(f"Saved to: {CHECKPOINT_DIR / 'best_league_champion.json'}")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kaggriculture Co-Evolution League")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_coevolution_league(generations=args.generations, population_size=args.population, steps=args.steps, seed=args.seed)
