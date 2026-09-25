"""Mixed-parameter Bayesian search for Kaggriculture policies.

Uses Optuna's Tree-structured Parzen Estimator (TPE) to choose promising
configuration values. Simulator access is local; the competition agent does
not import Optuna. Every trial is evaluated on the same training seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (BASELINE_PATH, HOLDOUT_SEEDS, PARAMETERS, TRAIN_SEEDS,
                    VALIDATION_SEEDS, load_config)
from experiments.evaluate import evaluate_opponent_pool


ROOT = Path(__file__).resolve().parents[1]


def suggest_config(trial: Any, base: dict[str, Any]) -> dict[str, Any]:
    """Ask an Optuna trial for each explicitly tunable policy parameter."""
    result = dict(base)
    for name, spec in PARAMETERS.items():
        kind = spec["type"]
        if kind == "float":
            result[name] = trial.suggest_float(name, spec["min"], spec["max"])
        elif kind == "int":
            result[name] = trial.suggest_int(name, spec["min"], spec["max"])
        elif kind == "bool":
            result[name] = trial.suggest_categorical(name, [False, True])
        elif kind == "categorical":
            result[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unsupported parameter type for {name}: {kind}")
    return result


def risk_adjusted_fitness(metrics: dict[str, Any], risk_weight: float) -> float:
    """Prefer high average margins while lightly penalizing volatile results."""
    return float(metrics["mean_margin"]) - risk_weight * float(metrics["pooled_std_margin"])


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["trial", "state", "fitness", "mean_margin", "std_margin",
              "mean_reward", "wins", "losses", "parameters"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _trial_params_for_config(config: dict[str, Any]) -> dict[str, Any]:
    return {name: config[name] for name in PARAMETERS}


def optimize_tpe(
    n_trials: int = 50,
    seed: int = 7,
    train_seeds: tuple[int, ...] = TRAIN_SEEDS,
    validation_seeds: tuple[int, ...] = VALIDATION_SEEDS,
    holdout_seeds: tuple[int, ...] = HOLDOUT_SEEDS,
    opponents: tuple[str, ...] = ("random",),
    steps: int = 720,
    finalists: int = 5,
    risk_weight: float = 0.10,
    seed_config_paths: tuple[Path, ...] = (),
    output_root: Path | None = None,
) -> Path:
    """Run reproducible TPE search, then validate finalists and test holdout.

    Holdout seeds are only read after the validation-selected config is fixed.
    The baseline file is evaluated but never changed.
    """
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is required for TPE search. Install it with: "
            "python -m pip install -r requirements-optimizer.txt"
        ) from exc
    if n_trials < 1 or finalists < 1:
        raise ValueError("n_trials and finalists must be positive")
    if not train_seeds or not validation_seeds or not holdout_seeds:
        raise ValueError("Training, validation, and holdout seed sets must be non-empty")
    if not opponents:
        raise ValueError("At least one opponent is required")

    output_root = output_root or ROOT / "experiments" / "results"
    run_dir = output_root / ("tpe_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    baseline = load_config(BASELINE_PATH)
    starting_configs = [load_config(path) for path in seed_config_paths]
    rows: list[dict[str, Any]] = []

    baseline_train = evaluate_opponent_pool(baseline, train_seeds, opponents, steps=steps)
    baseline_fitness = risk_adjusted_fitness(baseline_train, risk_weight)
    print(f"baseline train margin={baseline_train['mean_margin']:.2f} "
          f"fitness={baseline_fitness:.2f}", flush=True)

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, n_trials))
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                study_name=run_dir.name)
    for config in starting_configs:
        study.enqueue_trial(_trial_params_for_config(config))

    def objective(trial: Any) -> float:
        config = suggest_config(trial, baseline)
        metrics = evaluate_opponent_pool(config, train_seeds, opponents, steps=steps)
        fitness = risk_adjusted_fitness(metrics, risk_weight)
        trial.set_user_attr("config", config)
        trial.set_user_attr("training", metrics)
        trial.set_user_attr("fitness", fitness)
        print(f"trial={trial.number} margin={metrics['mean_margin']:.2f} "
              f"std={metrics['pooled_std_margin']:.2f} fitness={fitness:.2f}", flush=True)
        return fitness

    def checkpoint(current_study: Any, _: Any) -> None:
        trial_rows = []
        trial_reports = {}
        completed_trials = [item for item in current_study.trials
                            if item.state.name == "COMPLETE"]
        for item in current_study.trials:
            metrics = item.user_attrs.get("training", {})
            trial_rows.append({
                "trial": item.number,
                "state": item.state.name,
                "fitness": item.user_attrs.get("fitness", item.value),
                "mean_margin": metrics.get("mean_margin", ""),
                "std_margin": metrics.get("pooled_std_margin", ""),
                "mean_reward": metrics.get("mean_reward", ""),
                "wins": metrics.get("wins", ""),
                "losses": metrics.get("losses", ""),
                "parameters": json.dumps(item.user_attrs.get("config", item.params),
                                          sort_keys=True),
            })
            if item.state.name == "COMPLETE":
                trial_reports[str(item.number)] = {
                    "config": item.user_attrs["config"],
                    "training": item.user_attrs["training"],
                    "fitness": item.value,
                }
        _write_csv(run_dir / "trials.csv", trial_rows)
        _write_json(run_dir / "trial_reports.json", trial_reports)
        if completed_trials:
            best_trial = max(completed_trials, key=lambda item: float(item.value))
            _write_json(run_dir / "config_best_training.json",
                        best_trial.user_attrs["config"])

    study.optimize(objective, n_trials=n_trials, callbacks=[checkpoint],
                    catch=(KeyboardInterrupt,))
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not completed:
        raise RuntimeError("TPE search produced no completed trials")

    # Persist each trial's complete report, including trials that did not win.
    for trial in study.trials:
        metrics = trial.user_attrs.get("training", {})
        rows.append({
            "trial": trial.number,
            "state": trial.state.name,
            "fitness": trial.user_attrs.get("fitness", trial.value),
            "mean_margin": metrics.get("mean_margin", ""),
            "std_margin": metrics.get("pooled_std_margin", ""),
            "mean_reward": metrics.get("mean_reward", ""),
            "wins": metrics.get("wins", ""),
            "losses": metrics.get("losses", ""),
            "parameters": json.dumps(trial.user_attrs.get("config", trial.params), sort_keys=True),
        })
    _write_csv(run_dir / "trials.csv", rows)
    _write_json(run_dir / "config_best_training.json", study.best_trial.user_attrs["config"])

    ranked = sorted(completed, key=lambda trial: float(trial.value), reverse=True)
    candidate_trials = []
    seen: set[str] = set()
    for trial in ranked:
        config = trial.user_attrs["config"]
        key = json.dumps(config, sort_keys=True)
        if key not in seen:
            seen.add(key)
            candidate_trials.append(trial)
        if len(candidate_trials) >= finalists:
            break
    validation = {
        str(trial.number): evaluate_opponent_pool(
            trial.user_attrs["config"], validation_seeds, opponents, steps=steps)
        for trial in candidate_trials
    }
    selected = max(candidate_trials,
                   key=lambda trial: validation[str(trial.number)]["mean_margin"])
    best_config = selected.user_attrs["config"]

    # Holdout is evaluated only after selection has been made on validation.
    baseline_validation = evaluate_opponent_pool(baseline, validation_seeds,
                                                  opponents, steps=steps)
    baseline_holdout = evaluate_opponent_pool(baseline, holdout_seeds,
                                               opponents, steps=steps)
    best_validation = validation[str(selected.number)]
    best_holdout = evaluate_opponent_pool(best_config, holdout_seeds,
                                           opponents, steps=steps)
    best_training = selected.user_attrs["training"]
    report = {
        "algorithm": "Optuna TPE",
        "objective": "mean margin minus risk_weight times pooled standard deviation",
        "risk_weight": risk_weight,
        "opponents": list(opponents),
        "seed_sets": {"training": list(train_seeds), "validation": list(validation_seeds),
                      "holdout": list(holdout_seeds)},
        "baseline": {"training": baseline_train, "validation": baseline_validation,
                     "holdout": baseline_holdout},
        "optimized": {"trial": selected.number, "training": best_training,
                      "validation": best_validation, "holdout": best_holdout},
        "validation_finalists": {
            str(trial.number): {"training_fitness": trial.value,
                                "validation": validation[str(trial.number)]}
            for trial in candidate_trials
        },
        "possible_overfitting": (
            best_training["mean_margin"] > baseline_train["mean_margin"]
            and (best_validation["mean_margin"] < baseline_validation["mean_margin"]
                 or best_holdout["mean_margin"] < baseline_holdout["mean_margin"])
        ),
    }
    _write_json(run_dir / "config_best.json", best_config)
    _write_json(run_dir / "validation.json", report)
    _write_json(run_dir / "holdout.json", {
        "baseline": baseline_holdout, "optimized": best_holdout,
        "mean_margin_delta": best_holdout["mean_margin"] - baseline_holdout["mean_margin"],
    })
    print(json.dumps({key: report[key] for key in
                      ("algorithm", "seed_sets", "baseline", "optimized", "possible_overfitting")},
                     indent=2), flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=list(VALIDATION_SEEDS))
    parser.add_argument("--holdout-seeds", type=int, nargs="+", default=list(HOLDOUT_SEEDS))
    parser.add_argument("--opponents", nargs="+", default=["random"])
    parser.add_argument("--finalists", type=int, default=5)
    parser.add_argument("--risk-weight", type=float, default=0.10)
    parser.add_argument("--seed-config", type=Path, action="append", default=[])
    args = parser.parse_args()
    run = optimize_tpe(
        n_trials=args.trials, seed=args.seed,
        train_seeds=tuple(args.train_seeds),
        validation_seeds=tuple(args.validation_seeds),
        holdout_seeds=tuple(args.holdout_seeds), opponents=tuple(args.opponents),
        steps=args.steps, finalists=args.finalists, risk_weight=args.risk_weight,
        seed_config_paths=tuple(args.seed_config),
    )
    print(f"Results: {run}")


if __name__ == "__main__":
    main()
