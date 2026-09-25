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
from experiments.evaluate import evaluate_opponent_pool


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


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def optimize(population_size: int = 8, generations: int = 5, seed: int = 7,
             train_seeds: tuple[int, ...] = TRAIN_SEEDS,
             validation_seeds: tuple[int, ...] = VALIDATION_SEEDS,
             holdout_seeds: tuple[int, ...] = HOLDOUT_SEEDS,
             output_root: Path | None = None, steps: int = 720,
             opponent: str = "random",
             seed_config_paths: tuple[Path, ...] = (),
             opponents: tuple[str, ...] | None = None) -> Path:
    rng = random.Random(seed)
    output_root = output_root or Path(__file__).resolve().parents[1] / "experiments" / "results"
    run_dir = output_root / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline = load_config()
    opponent_pool = tuple(opponents or (opponent,))
    population = [dict(baseline)]
    for path in seed_config_paths:
        if len(population) >= population_size:
            break
        population.append(load_config(path))
    while len(population) < population_size:
        parent = rng.choice(population)
        population.append(mutate(parent, rng, rate=0.45))
    evaluations: dict[str, dict[str, Any]] = {}
    candidate_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    candidate_num = 0
    baseline_score = evaluate_opponent_pool(baseline, train_seeds, opponent_pool, steps=steps)
    print(f"baseline training_margin={baseline_score['mean_margin']:.3f}", flush=True)
    for generation in range(generations):
        scored = []
        for candidate in population:
            candidate_id = f"c{candidate_num:04d}"
            candidate_num += 1
            metrics = evaluate_opponent_pool(candidate, train_seeds, opponent_pool, steps=steps)
            evaluations[candidate_id] = {"config": candidate, "training": metrics}
            # The competition objective is relative wealth: maximize margin
            # against the opponent, not just our own cash. A candidate can
            # earn more while helping the opponent earn even more.
            fitness = metrics["mean_margin"]
            scored.append((fitness, candidate_id, candidate))
            print(f"generation={generation} candidate={candidate_id} "
                  f"training_margin={fitness:.3f}", flush=True)
            row = {"generation": generation, "candidate_id": candidate_id,
                   "fitness": fitness, "validation_score": "",
                   "parameters": json.dumps(candidate, sort_keys=True)}
            candidate_rows.append(row)
            _append_csv(run_dir / "population.csv",
                        ["generation", "candidate_id", "fitness", "validation_score", "parameters"],
                        row)
            best_so_far = max(evaluations.values(), key=lambda result: result["training"]["mean_margin"])
            _write_json(run_dir / "config_best_training.json", best_so_far["config"])
        scored.sort(key=lambda row: row[0], reverse=True)
        generation_rows.append({"generation": generation,
                                "best_fitness": scored[0][0],
                                "mean_fitness": sum(x[0] for x in scored) / len(scored),
                                "baseline_fitness": baseline_score["mean_margin"]})
        # Flush progress every generation so long searches leave usable records
        # if they are stopped before validation and holdout complete.
        _write_csv(run_dir / "population.csv",
                   ["generation", "candidate_id", "fitness", "validation_score", "parameters"],
                   candidate_rows)
        _write_csv(run_dir / "generation_metrics.csv",
                   ["generation", "best_fitness", "mean_fitness", "baseline_fitness"],
                   generation_rows)
        best_so_far = max(evaluations.values(), key=lambda result: result["training"]["mean_margin"])
        _write_json(run_dir / "config_best_training.json", best_so_far["config"])
        elite_n = max(1, population_size // 4)
        next_population = [dict(row[2]) for row in scored[:elite_n]]
        while len(next_population) < population_size:
            parent_a = rng.choice(scored[:max(2, len(scored) // 2)])[2]
            parent_b = rng.choice(scored[:max(2, len(scored) // 2)])[2]
            next_population.append(mutate(crossover(parent_a, parent_b, rng), rng))
        population = next_population

    ranked = sorted(evaluations.items(), key=lambda pair: pair[1]["training"]["mean_margin"], reverse=True)
    finalist_ids = [candidate_id for candidate_id, _ in ranked[:min(3, len(ranked))]]
    reference_matches: list[tuple[str, dict[str, Any], str | None]] = []
    reference_configs = [(path.stem, load_config(path)) for path in seed_config_paths]
    for reference_name, reference_config in reference_configs:
        match = next((candidate_id for candidate_id, result in evaluations.items()
                      if result["config"] == reference_config), None)
        if match is not None and match not in finalist_ids:
            finalist_ids.append(match)
        reference_matches.append((reference_name, reference_config, match))
    validation_scores = {}
    for candidate_id in finalist_ids:
        validation_scores[candidate_id] = evaluate_opponent_pool(
            evaluations[candidate_id]["config"], validation_seeds,
            opponent_pool, steps=steps)
        print(f"validation candidate={candidate_id} "
              f"mean_margin={validation_scores[candidate_id]['mean_margin']:.3f}", flush=True)
        for row in candidate_rows:
            if row["candidate_id"] == candidate_id:
                row["validation_score"] = validation_scores[candidate_id]["mean_margin"]
    baseline_validation = evaluate_opponent_pool(baseline, validation_seeds,
                                                  opponent_pool, steps=steps)
    selected_id = max(finalist_ids, key=lambda cid: validation_scores[cid]["mean_margin"])
    best_config = evaluations[selected_id]["config"]
    baseline_holdout = evaluate_opponent_pool(baseline, holdout_seeds,
                                               opponent_pool, steps=steps)
    best_holdout = evaluate_opponent_pool(best_config, holdout_seeds,
                                           opponent_pool, steps=steps)
    print("holdout evaluation complete", flush=True)
    best_training = evaluations[selected_id]["training"]
    best_validation = validation_scores[selected_id]
    starting_policy_reports = {}
    for name, reference_config, reference_id in reference_matches:
        reference_training = (evaluations[reference_id]["training"] if reference_id else
                              evaluate_opponent_pool(reference_config, train_seeds,
                                                     opponent_pool, steps=steps))
        reference_validation = (validation_scores[reference_id] if reference_id else
                                evaluate_opponent_pool(reference_config, validation_seeds,
                                                       opponent_pool, steps=steps))
        reference_holdout = (best_holdout if reference_config == best_config else
                             evaluate_opponent_pool(reference_config, holdout_seeds,
                                                    opponent_pool, steps=steps))
        starting_policy_reports[name] = {
            "candidate_id": reference_id, "training": reference_training,
            "validation": reference_validation, "holdout": reference_holdout,
        }
    comparison_reference = (next(iter(starting_policy_reports.values()))
                           if starting_policy_reports else
                           {"training": baseline_score, "validation": baseline_validation,
                            "holdout": baseline_holdout})
    report = {
        "opponents": list(opponent_pool),
        "selection_metric": "mean_margin averaged equally across opponents",
        "seed_sets": {"training": list(train_seeds), "validation": list(validation_seeds),
                      "holdout": list(holdout_seeds)},
        "baseline": {"training": baseline_score, "validation": baseline_validation,
                     "holdout": baseline_holdout},
        "starting_policies": starting_policy_reports,
        "optimized": {"candidate_id": selected_id, "training": best_training,
                      "validation": best_validation, "holdout": best_holdout},
        "possible_overfitting": (
            best_training["mean_margin"] > comparison_reference["training"]["mean_margin"]
            and best_holdout["mean_margin"] < comparison_reference["holdout"]["mean_margin"]),
    }
    _write_json(run_dir / "config_best.json", best_config)
    _write_json(run_dir / "validation.json", report)
    _write_json(run_dir / "holdout.json", {"baseline": baseline_holdout, "optimized": best_holdout})
    _write_csv(run_dir / "population.csv",
               ["generation", "candidate_id", "fitness", "validation_score", "parameters"],
               candidate_rows)
    _write_csv(run_dir / "generation_metrics.csv",
               ["generation", "best_fitness", "mean_fitness", "baseline_fitness"],
               generation_rows)
    print(json.dumps(report, indent=2))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=None,
                        help="fixed seed set shared by every candidate (defaults to the project's training set)")
    parser.add_argument("--opponent", default="random",
                        help="local Kaggriculture opponent agent name, e.g. random or starter")
    parser.add_argument("--opponents", nargs="+",
                        help="evaluate against this opponent pool; accepts random, starter, or replay:<path>:<player>")
    parser.add_argument("--seed-config", type=Path, action="append", default=[],
                        help="include a hand-built starting candidate (repeatable)")
    args = parser.parse_args()
    if args.opponents and args.opponent != "random":
        parser.error("use either --opponent or --opponents, not both")
    optimize(args.population, args.generations, args.seed, steps=args.steps,
             opponent=args.opponent,
             opponents=tuple(args.opponents) if args.opponents else None,
             train_seeds=tuple(args.train_seeds) if args.train_seeds else TRAIN_SEEDS,
             seed_config_paths=tuple(args.seed_config))


if __name__ == "__main__":
    main()
