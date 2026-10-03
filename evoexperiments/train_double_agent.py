"""Double-Agent Co-Evolutionary Training System for Kaggriculture.

Features:
1. Two distinct co-evolving populations (Alpha: Livestock/Fertilizer, Beta: High-Margin Crops/Counter-Trading).
2. Direct head-to-head self-play matches sharing the dynamic market.
3. Multi-Agent Live League: CompetingAgent + SwarmPlanterAgent + Starter across dynamic seeds.
4. Auto-export and promotion to submission.py and submission.tar.gz on new records.
5. Multi-hour autonomous duration support with high-throughput multiprocessing.
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

from evoexperiments.double_agent_engine import (
    DEFAULT_GENOME,
    GENOME_SPECS,
    DoubleAgentPolicy,
)
from experiments.opponents import seeded_random_agent
from kaggle_environments import make

CHECKPOINT_DIR = ROOT / "evoexperiments" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def mutate_genome(genome: Dict[str, Any], rng: random.Random, rate: float = 0.30) -> Dict[str, Any]:
    child = dict(genome)
    for key, spec in GENOME_SPECS.items():
        if rng.random() >= rate:
            continue
        kind = spec["type"]
        if kind == "int":
            span = max(1, (spec["max"] - spec["min"]) // 4)
            child[key] = max(spec["min"], min(spec["max"], int(child[key]) + rng.randint(-span, span)))
        elif kind == "float":
            span = (spec["max"] - spec["min"]) * 0.20
            val = float(child[key]) + rng.uniform(-span, span)
            child[key] = round(max(spec["min"], min(spec["max"], val)), 2)
    return child


def crossover_genomes(parent1: Dict[str, Any], parent2: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    child = {}
    for key in DEFAULT_GENOME.keys():
        child[key] = parent1[key] if rng.random() < 0.5 else parent2.get(key, parent1[key])
    return child


def run_single_game(args: Tuple[Dict[str, Any], Any, int, int]) -> Dict[str, Any]:
    agent_cfg, opp_type_or_data, seed, steps = args
    bot0 = DoubleAgentPolicy(agent_cfg)

    if opp_type_or_data == "starter":
        opp_bot = "starter"
    elif opp_type_or_data == "random":
        opp_bot = seeded_random_agent(seed)
    elif opp_type_or_data == "competing":
        from evoexperiments.competing_agent import CompetingAgent
        bot1 = CompetingAgent()
        opp_bot = lambda obs: bot1(obs)
    elif opp_type_or_data == "swarm":
        from evoexperiments.league_agents import SwarmPlanterAgent
        bot1 = SwarmPlanterAgent()
        opp_bot = lambda obs: bot1(obs)
    elif opp_type_or_data == "dairy":
        from evoexperiments.league_agents import DairyBaronAgent
        bot1 = DairyBaronAgent()
        opp_bot = lambda obs: bot1(obs)
    elif isinstance(opp_type_or_data, dict):
        bot1 = DoubleAgentPolicy(opp_type_or_data)
        opp_bot = lambda obs: bot1(obs)
    else:
        opp_bot = "starter"

    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    env.run([lambda obs: bot0(obs), opp_bot if isinstance(opp_bot, str) else (lambda obs: opp_bot(obs))])

    final = env.steps[-1]
    p0 = float(final[0].reward or 0)
    p1 = float(final[1].reward or 0)
    margin = p0 - p1
    return {
        "seed": seed,
        "opp": str(opp_type_or_data) if not isinstance(opp_type_or_data, dict) else "self_play",
        "p0_reward": p0,
        "p1_reward": p1,
        "margin": margin,
        "win": 1 if margin > 0 else 0,
    }


def evaluate_candidate(
    candidate: Dict[str, Any],
    peer_candidate: Dict[str, Any],
    train_seeds: Tuple[int, ...],
    steps: int = 720,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Dict[str, Any]:
    tasks = []

    # 1. Head-to-Head Self-Play (Shared Market Arena) across 2 dynamic seeds
    tasks.append((candidate, peer_candidate, train_seeds[0], steps))
    tasks.append((candidate, peer_candidate, train_seeds[1], steps))

    # 2. Live League Matches
    tasks.append((candidate, "competing", train_seeds[0], steps))
    tasks.append((candidate, "swarm", train_seeds[1], steps))
    tasks.append((candidate, "starter", train_seeds[2], steps))

    if executor:
        results = list(executor.map(run_single_game, tasks))
    else:
        results = [run_single_game(t) for t in tasks]

    self_play_res = results[:2]
    league_res = results[2:]

    sp_money = sum(r["p0_reward"] for r in self_play_res) / len(self_play_res)
    sp_margin = sum(r["margin"] for r in self_play_res) / len(self_play_res)

    league_money = sum(r["p0_reward"] for r in league_res) / len(league_res)
    league_margin = sum(r["margin"] for r in league_res) / len(league_res)
    league_wins = sum(r["win"] for r in league_res)

    # Multi-Objective Fitness: rewards both league performance and self-play dominance
    fitness = league_money + 0.8 * league_margin + 0.5 * sp_money + 0.5 * sp_margin + (5000 * league_wins)

    return {
        "fitness": fitness,
        "sp_money": sp_money,
        "sp_margin": sp_margin,
        "league_money": league_money,
        "league_margin": league_margin,
        "league_wins": league_wins,
        "results": results,
    }


def train_double_agents(
    duration_hours: float = 2.0,
    pop_size: int = 4,
    steps: int = 720,
    seed: int = 42,
):
    rng = random.Random(seed)

    # Initial Population Alpha (Central Ring Livestock / Feed tuning)
    alpha_base = dict(DEFAULT_GENOME)
    alpha_base.update({"base_sheep_goal": 4, "base_cow_goal": 5, "base_goose_goal": 4, "drop_threshold": 2})

    # Initial Population Beta (Outer Ring Cash Crops / Strawberries & Market tuning)
    beta_base = dict(DEFAULT_GENOME)
    beta_base.update({"melon_weight": 2.2, "strawberry_weight": 2.2, "sell_price_ratio": 0.16})

    best_overall_fitness = -float("inf")
    best_overall_money = 0.0
    best_overall_genome = dict(alpha_base)

    # Resume from checkpoint if it exists and matches schema
    checkpoint_file = CHECKPOINT_DIR / "best_double_agent.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                prev_data = json.load(f)
            raw_genome = prev_data.get("genome", {})
            valid_genome = {k: raw_genome[k] for k in DEFAULT_GENOME.keys() if k in raw_genome}
            if len(valid_genome) >= len(DEFAULT_GENOME) - 2:
                alpha_base.update(valid_genome)
                best_overall_fitness = prev_data.get("best_fitness", -float("inf"))
                best_overall_money = prev_data.get("best_money", 0.0)
                best_overall_genome = dict(alpha_base)
                print(f"  --> [RESUMING FROM CHECKPOINT] Peak Money=${best_overall_money:+,.0f} | Fitness={best_overall_fitness:,.0f}")
            else:
                print("  --> [INITIALIZING FRESH FROM OPTIMAL GRANDMASTER BASE]")
        except Exception as e:
            print(f"Warning loading checkpoint: {e}")

    alpha_pop = [dict(alpha_base)] + [mutate_genome(alpha_base, rng, 0.35) for _ in range(pop_size - 1)]
    beta_pop = [dict(beta_base)] + [mutate_genome(beta_base, rng, 0.35) for _ in range(pop_size - 1)]

    champ_alpha = dict(alpha_base)
    champ_beta = dict(beta_base)

    workers = max(1, min((os.cpu_count() or 2) - 1, 7))
    start_time = time.time()
    max_duration_seconds = duration_hours * 3600

    print("=" * 80)
    print(f"STARTING 2-HOUR AUTONOMOUS DOUBLE-AGENT HYPER-TRAINING")
    print(f"Target Duration: {duration_hours:.1f} hours ({max_duration_seconds:,.0f}s) | Pop Size: {pop_size}x2 | Workers: {workers}")
    print(f"Arena: Shared Market Self-Play + Live League (CompetingAgent, SwarmPlanter, Starter)")
    print("=" * 80)

    gen = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= max_duration_seconds:
                print(f"\n[TARGET DURATION OF {duration_hours} HOURS REACHED!]")
                break

            gen += 1
            gen_t0 = time.time()
            rem = max(0.0, max_duration_seconds - elapsed)

            # Evolve seeds dynamically so agents learn cross-seed generalization
            gen_seed_base = (seed + gen * 43) % 900_000 + 1000
            train_seeds = (gen_seed_base, gen_seed_base + 1, gen_seed_base + 2)

            print(f"\n--- Generation {gen} [Elapsed: {elapsed/60:.1f}m / {duration_hours*60:.0f}m | Left: {rem/60:.1f}m] ---")

            # 1. Evaluate Alpha Population against Beta Champion
            alpha_scored = []
            for cand in alpha_pop:
                score = evaluate_candidate(cand, champ_beta, train_seeds, steps, executor)
                alpha_scored.append((score["fitness"], score["league_money"], score["league_margin"], cand))

            alpha_scored.sort(key=lambda x: -x[0])
            best_a_fit, best_a_money, best_a_margin, champ_alpha = alpha_scored[0]
            print(f"  Alpha Top: LeagueMoney=${best_a_money:+,.0f} | Margin=${best_a_margin:+,.0f} | Fit={best_a_fit:,.0f}")

            # 2. Evaluate Beta Population against Alpha Champion
            beta_scored = []
            for cand in beta_pop:
                score = evaluate_candidate(cand, champ_alpha, train_seeds, steps, executor)
                beta_scored.append((score["fitness"], score["league_money"], score["league_margin"], cand))

            beta_scored.sort(key=lambda x: -x[0])
            best_b_fit, best_b_money, best_b_margin, champ_beta = beta_scored[0]
            print(f"  Beta  Top: LeagueMoney=${best_b_money:+,.0f} | Margin=${best_b_margin:+,.0f} | Fit={best_b_fit:,.0f}")

            # 3. Check for Global Champion
            gen_top = alpha_scored[0] if alpha_scored[0][0] >= beta_scored[0][0] else beta_scored[0]
            if gen_top[0] > best_overall_fitness:
                best_overall_fitness = gen_top[0]
                best_overall_money = gen_top[1]
                best_overall_genome = dict(gen_top[3])
                print(f"  🏆 [NEW RECORD CHAMPION FOUND!] LeagueMoney=${best_overall_money:+,.0f} | Fitness={best_overall_fitness:,.0f}")

                # Save Checkpoint
                champ_payload = {
                    "generation": gen,
                    "elapsed_minutes": elapsed / 60.0,
                    "best_fitness": best_overall_fitness,
                    "best_money": best_overall_money,
                    "genome": best_overall_genome,
                }
                with open(CHECKPOINT_DIR / "best_double_agent.json", "w") as f:
                    json.dump(champ_payload, f, indent=2)

                try:
                    from evoexperiments.export_champion import export
                    export()
                except Exception as e:
                    print(f"Auto-export notification: {e}")

            # 4. Reproduce & Evolve Alpha Population
            new_alpha = [dict(champ_alpha)]
            while len(new_alpha) < pop_size:
                p1 = rng.choice(alpha_scored[:2])[3]
                p2 = rng.choice(alpha_scored[:3])[3]
                child = crossover_genomes(p1, p2, rng)
                new_alpha.append(mutate_genome(child, rng, 0.30))
            alpha_pop = new_alpha

            # 5. Reproduce & Evolve Beta Population
            new_beta = [dict(champ_beta)]
            while len(new_beta) < pop_size:
                p1 = rng.choice(beta_scored[:2])[3]
                p2 = rng.choice(beta_scored[:3])[3]
                child = crossover_genomes(p1, p2, rng)
                new_beta.append(mutate_genome(child, rng, 0.30))
            beta_pop = new_beta

            dt = time.time() - gen_t0
            print(f"  Gen {gen} finished in {dt:.1f}s")

    print("\n" + "=" * 80)
    print("HYPER-TRAINING SESSION COMPLETED SUCCESSFULLY!")
    print(f"Total Generations: {gen}")
    print(f"All-Time Peak Champion: Money=${best_overall_money:+,.0f} | Fitness={best_overall_fitness:,.0f}")
    print("Checkpoints saved and latest champion exported to submission.py & submission.tar.gz")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration_hours", type=float, default=2.0)
    parser.add_argument("--pop_size", type=int, default=4)
    args = parser.parse_args()
    train_double_agents(duration_hours=args.duration_hours, pop_size=args.pop_size)
