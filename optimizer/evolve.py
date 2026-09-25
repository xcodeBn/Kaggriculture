"""Small reproducible genetic search over explicit policy parameters."""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (DEFAULT_CONFIG, HOLDOUT_SEEDS, PARAMETERS, TRAIN_SEEDS,
                    VALIDATION_SEEDS, load_config)
from experiments.evaluate import evaluate_config


def random_value(spec: dict[str, Any], rng: random.Random) -> Any:
    kind = spec["type"]
    if kind == "float":
        return rng.uniform(spec["min"], spec["max"])
    if kind == "int":
        return rng.randint(spec["min"], spec["max"])
    if kind == "bool":
        return bool(rng.getrandbits(1))
    if kind == "categorical":
        return rng.choice(spec["choices"])
    raise ValueError(f"Unsupported parameter type: {kind}")


def mutate(config: dict[str, Any], rng: random.Random,
           rate: float = 0.2) -> dict[str, Any]:
    child = dict(config)
    for key, spec in PARAMETERS.items():
        if rng.random() >= rate:
            continue
        kind = spec["type"]
        if kind in ("bool", "categorical"):
            child[key] = random_value(spec, rng)
        elif kind == "int":
            span = max(1, (spec["max"] - spec["min"]) // 5)
            child[key] = max(spec["min"], min(spec["max"], int(child[key]) + rng.randint(-span, span)))
        elif kind == "float":
            span = (spec["max"] - spec["min"]) * 0.15
            child[key] = max(spec["min"], min(spec["max"], float(child[key]) + rng.uniform(-span, span)))
    return child


def crossover(left: dict[str, Any], right: dict[str, Any],
              rng: random.Random) -> dict[str, Any]:
    child = dict(left)
    for key in PARAMETERS:
        if rng.random() < 0.5:
            child[key] = right[key]
    return child


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def optimize(population_size: int = 8, generations: int = 5, seed: int = 7,
             train_seeds: tuple[int, ...] = TRAIN_SEEDS,
             validation_seeds: tuple[int, ...] = VALIDATION_SEEDS,
             holdout_seeds: tuple[int, ...] = HOLDOUT_SEEDS,
             output_root: Path | None = None, steps: int = 720) -> Path:
    rng = random.Random(seed)
    output_root = output_root or Path(__file__).resolve().parents[1] / "experiments" / "results"
    run_dir = output_root / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline = load_config()
    population = [dict(baseline)] + [mutate(baseline, rng, rate=0.45)
                                      for _ in range(population_size - 1)]
    evaluations: dict[str, dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    candidate_num = 0
    baseline_score = evaluate_config(baseline, train_seeds, steps=steps)
    for generation in range(generations):
        scored = []
        for candidate in population:
            candidate_id = f"c{candidate_num:04d}"
            candidate_num += 1
            metrics = evaluate_config(candidate, train_seeds, steps=steps)
            evaluations[candidate_id] = {"config": candidate, "training": metrics}
            scored.append((metrics["mean_reward"], candidate_id, candidate))
            print(f"generation={generation} candidate={candidate_id} "
                  f"training_mean={metrics['mean_reward']:.3f}", flush=True)
            candidate_rows.append({"generation": generation, "candidate_id": candidate_id,
                                   "fitness": metrics["mean_reward"],
                                   "validation_score": "",
                                   "parameters": json.dumps(candidate, sort_keys=True)})
        scored.sort(key=lambda row: row[0], reverse=True)
        generation_rows.append({"generation": generation,
                                "best_fitness": scored[0][0],
                                "mean_fitness": sum(x[0] for x in scored) / len(scored),
                                "baseline_fitness": baseline_score["mean_reward"]})
        elite_n = max(1, population_size // 4)
        next_population = [dict(row[2]) for row in scored[:elite_n]]
        while len(next_population) < population_size:
            parent_a = rng.choice(scored[:max(2, len(scored) // 2)])[2]
            parent_b = rng.choice(scored[:max(2, len(scored) // 2)])[2]
            next_population.append(mutate(crossover(parent_a, parent_b, rng), rng))
        population = next_population

    ranked = sorted(evaluations.items(), key=lambda pair: pair[1]["training"]["mean_reward"], reverse=True)
    finalist_ids = [candidate_id for candidate_id, _ in ranked[:min(3, len(ranked))]]
    validation_scores = {}
    for candidate_id in finalist_ids:
        validation_scores[candidate_id] = evaluate_config(
            evaluations[candidate_id]["config"], validation_seeds, steps=steps)
        print(f"validation candidate={candidate_id} "
              f"mean={validation_scores[candidate_id]['mean_reward']:.3f}", flush=True)
        for row in candidate_rows:
            if row["candidate_id"] == candidate_id:
                row["validation_score"] = validation_scores[candidate_id]["mean_reward"]
    baseline_validation = evaluate_config(baseline, validation_seeds, steps=steps)
    selected_id = max(finalist_ids, key=lambda cid: validation_scores[cid]["mean_reward"])
    best_config = evaluations[selected_id]["config"]
    baseline_holdout = evaluate_config(baseline, holdout_seeds, steps=steps)
    best_holdout = evaluate_config(best_config, holdout_seeds, steps=steps)
    print("holdout evaluation complete", flush=True)
    best_training = evaluations[selected_id]["training"]
    best_validation = validation_scores[selected_id]
    report = {
        "seed_sets": {"training": list(train_seeds), "validation": list(validation_seeds),
                      "holdout": list(holdout_seeds)},
        "baseline": {"training": baseline_score, "validation": baseline_validation,
                     "holdout": baseline_holdout},
        "optimized": {"candidate_id": selected_id, "training": best_training,
                      "validation": best_validation, "holdout": best_holdout},
        "possible_overfitting": (
            best_training["mean_reward"] > baseline_score["mean_reward"]
            and (best_validation["mean_reward"] < baseline_validation["mean_reward"]
                 or best_holdout["mean_reward"] < baseline_holdout["mean_reward"])),
    }
    _write_json(run_dir / "config_best.json", best_config)
    _write_json(run_dir / "validation.json", report)
    _write_json(run_dir / "holdout.json", {"baseline": baseline_holdout, "optimized": best_holdout})
    with (run_dir / "population.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["generation", "candidate_id", "fitness",
                                                     "validation_score", "parameters"])
        writer.writeheader(); writer.writerows(candidate_rows)
    with (run_dir / "generation_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["generation", "best_fitness", "mean_fitness", "baseline_fitness"])
        writer.writeheader(); writer.writerows(generation_rows)
    print(json.dumps(report, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=720)
    args = parser.parse_args()
    optimize(args.population, args.generations, args.seed, steps=args.steps)


if __name__ == "__main__":
    main()
