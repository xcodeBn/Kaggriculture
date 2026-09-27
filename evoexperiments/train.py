"""Fast Recursive Evolutionary Training Harness for Kaggriculture.

Features:
1. Multi-Core Acceleration: Uses parallel_eval to evaluate full generations in seconds.
2. Recursive Iteration: Starts from current best config (optimized_config.json or custom seed),
   iteratively evolves, validates on separate seeds, and checkpoints the new champion.
3. Multi-Opponent Stress Pool: Tests candidates against diverse opponents (replay bots,
   starter, and random) to prevent overfitting to a single play style.
4. Auto-Saving & Resumable: Saves the current champion after every generation to
   evoexperiments/checkpoints/best_config.json.
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
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    DEFAULT_CONFIG,
    HOLDOUT_SEEDS,
    PARAMETERS,
    TRAIN_SEEDS,
    VALIDATION_SEEDS,
    load_config,
)
from evoexperiments.parallel_eval import parallel_evaluate_pool


CHECKPOINT_DIR = ROOT / "evoexperiments" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def mutate(config: Dict[str, Any], rng: random.Random, rate: float = 0.25) -> Dict[str, Any]:
    """Mutate configuration within bounded parameter space."""
    child = dict(config)
    for key, spec in PARAMETERS.items():
        if key not in child:
            continue
        if rng.random() >= rate:
            continue
        kind = spec["type"]
        if kind == "bool":
            child[key] = bool(rng.getrandbits(1))
        elif kind == "categorical":
            child[key] = rng.choice(spec["choices"])
        elif kind == "int":
            span = max(1, (spec["max"] - spec["min"]) // 6)
            child[key] = max(spec["min"], min(spec["max"], int(child[key]) + rng.randint(-span, span)))
        elif kind == "float":
            span = (spec["max"] - spec["min"]) * 0.12
            child[key] = max(spec["min"], min(spec["max"], float(child[key]) + rng.uniform(-span, span)))
    return child


def crossover(left: Dict[str, Any], right: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Uniform crossover between two configs."""
    child = dict(left)
    for key in left:
        if key in right and rng.random() < 0.5:
            child[key] = right[key]
    return child


def run_recursive_training(
    generations: int = 10,
    population_size: int = 6,
    seed: int = 42,
    train_seeds: Tuple[int, ...] = (1000, 1001, 1002),
    val_seeds: Tuple[int, ...] = (2000, 2001),
    opponents: Tuple[str, ...] = ("random", "starter"),
    seed_config_path: Path | None = None,
    steps: int = 720
) -> Dict[str, Any]:
    """Execute multi-generation recursive evolutionary training."""
    rng = random.Random(seed)
    
    # Load initial seed configuration
    initial_path = seed_config_path or (ROOT / "optimized_config.json")
    champion = load_config(initial_path)
    print(f"\n=======================================================")
    print(f"STARTING RECURSIVE TRAINING: {generations} Generations")
    print(f"Population: {population_size} | Steps/game: {steps}")
    print(f"Opponents: {list(opponents)}")
    print(f"Initial seed config: {initial_path.name}")
    print(f"=======================================================\n")

    # Evaluate initial champion
    print("Evaluating initial champion on validation seeds...")
    val_res = parallel_evaluate_pool(champion, val_seeds, opponents, steps=steps)
    best_val_margin = val_res["mean_margin"]
    print(f"Initial Champion Validation Margin: ${best_val_margin:+,.0f} (Wins: {val_res['wins']}/{val_res['games']})\n")

    # Seed population
    population = [dict(champion)]
    while len(population) < population_size:
        population.append(mutate(champion, rng, rate=0.35))

    history = []

    for gen in range(1, generations + 1):
        t0 = time.time()
        print(f"--- Generation {gen}/{generations} ---")

        # 1. Parallel evaluation of all candidates on training seeds
        scores = []
        for idx, candidate in enumerate(population):
            eval_res = parallel_evaluate_pool(candidate, train_seeds, opponents, steps=steps)
            margin = eval_res["mean_margin"]
            scores.append((margin, candidate))

        # Sort descending by training margin
        scores.sort(key=lambda x: x[0], reverse=True)
        top_margin, top_candidate = scores[0]
        dt = time.time() - t0

        print(f"Gen {gen} Train Best Margin: ${top_margin:+,.0f} (Completed in {dt:.1f}s)")

        # 2. Validate top candidate on separate validation seeds
        val_check = parallel_evaluate_pool(top_candidate, val_seeds, opponents, steps=steps)
        candidate_val_margin = val_check["mean_margin"]
        print(f"Gen {gen} Validation Margin: ${candidate_val_margin:+,.0f} vs Champion: ${best_val_margin:+,.0f}")

        # 3. Check for improvement
        if candidate_val_margin > best_val_margin:
            gain = candidate_val_margin - best_val_margin
            print(f"🔥 NEW CHAMPION DISCOVERED! (+${gain:,.0f} improvement)")
            champion = dict(top_candidate)
            best_val_margin = candidate_val_margin

            # Save checkpoint
            ckpt_path = CHECKPOINT_DIR / "best_config.json"
            ckpt_path.write_text(json.dumps(champion, indent=2), encoding="utf-8")
            gen_ckpt = CHECKPOINT_DIR / f"champion_gen_{gen}.json"
            gen_ckpt.write_text(json.dumps(champion, indent=2), encoding="utf-8")
            print(f"Saved checkpoint to {ckpt_path.relative_to(ROOT)}")
        else:
            print(f"Champion retained (${best_val_margin:+,.0f})")

        history.append({
            "gen": gen,
            "train_best_margin": top_margin,
            "val_margin": candidate_val_margin,
            "champion_val_margin": best_val_margin,
        })

        # 4. Next generation reproduction
        survivors = [c for _, c in scores[:max(2, population_size // 2)]]
        next_population = [dict(champion)] # Elitism: keep current champion
        
        while len(next_population) < population_size:
            p1 = rng.choice(survivors)
            p2 = rng.choice(survivors)
            child = crossover(p1, p2, rng)
            child = mutate(child, rng, rate=0.25)
            next_population.append(child)

        population = next_population
        print()

    print(f"=======================================================")
    print(f"TRAINING COMPLETE! Best Validation Margin: ${best_val_margin:+,.0f}")
    print(f"Final champion saved at: evoexperiments/checkpoints/best_config.json")
    print(f"=======================================================\n")
    return champion


def main():
    parser = argparse.ArgumentParser(description="Run recursive evolutionary training.")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations to evolve.")
    parser.add_argument("--population", type=int, default=6, help="Population size.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument("--steps", type=int, default=720, help="Steps per game (default 720).")
    parser.add_argument("--opponents", nargs="+", default=["random", "starter"],
                        help="List of opponents to train against (e.g. random starter replay:replays/kaggle_replays/113441225.json:1)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test with 120 steps.")
    args = parser.parse_args()

    steps = 120 if args.quick else args.steps
    run_recursive_training(
        generations=args.generations,
        population_size=args.population,
        seed=args.seed,
        steps=steps,
        opponents=tuple(args.opponents)
    )


if __name__ == "__main__":
    main()
