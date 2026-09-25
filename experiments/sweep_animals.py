"""Focused, fixed-seed sweep over livestock capacity and land expansion."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import HOLDOUT_SEEDS, VALIDATION_SEEDS, load_config
from experiments.evaluate import evaluate_config


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def sweep(base_config: dict[str, Any], counts: tuple[int, ...], species: tuple[str, ...],
          train_seeds: tuple[int, ...], validation_seeds: tuple[int, ...],
          holdout_seeds: tuple[int, ...], opponent: str, output_root: Path,
          finalist_count: int = 3) -> Path:
    output = output_root / ("animal_sweep_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    baseline = load_config()
    baseline_training = evaluate_config(baseline, train_seeds, opponent=opponent)
    candidates = []
    for animal in species:
        for count in counts:
            if count == 0:
                continue
            for expand in (False, True):
                cfg = dict(base_config, max_animals=count, animal_species=animal,
                           sheep_goal=0, cow_goal=0, buy_land_enabled=expand,
                           max_hands=max(8, int(base_config.get("max_hands", 0))),
                           min_cash_for_hand=0)
                candidate_id = f"{animal.lower()}_{count}_land{int(expand)}"
                metrics = evaluate_config(cfg, train_seeds, opponent=opponent)
                candidates.append({"candidate_id": candidate_id, "config": cfg,
                                   "training": metrics})
                print(f"candidate={candidate_id} training_margin={metrics['mean_margin']:.1f}",
                      flush=True)

    ranked = sorted(candidates, key=lambda item: item["training"]["mean_margin"],
                    reverse=True)
    finalists = ranked[:min(finalist_count, len(ranked))]
    starting = next((item for item in candidates if item["config"] == base_config), None)
    if starting is not None and starting not in finalists:
        finalists.append(starting)
    elif starting is None:
        starting = {"candidate_id": "starting_config", "config": base_config,
                    "training": evaluate_config(base_config, train_seeds,
                                                opponent=opponent)}
    for candidate in finalists:
        candidate["validation"] = evaluate_config(candidate["config"], validation_seeds,
                                                  opponent=opponent)
        print(f"validation candidate={candidate['candidate_id']} "
              f"margin={candidate['validation']['mean_margin']:.1f}", flush=True)
    selected = max(finalists, key=lambda item: item["validation"]["mean_margin"])
    baseline_validation = evaluate_config(baseline, validation_seeds, opponent=opponent)
    baseline_holdout = evaluate_config(baseline, holdout_seeds, opponent=opponent)
    selected_holdout = evaluate_config(selected["config"], holdout_seeds,
                                       opponent=opponent)
    starting_validation = starting.get("validation") or evaluate_config(
        base_config, validation_seeds, opponent=opponent)
    starting_holdout = (selected_holdout if selected["config"] == base_config else
                        evaluate_config(base_config, holdout_seeds, opponent=opponent))
    report = {
        "opponent": opponent,
        "selection_metric": "mean_margin",
        "seed_sets": {"training": list(train_seeds), "validation": list(validation_seeds),
                      "holdout": list(holdout_seeds)},
        "baseline": {"training": baseline_training, "validation": baseline_validation,
                     "holdout": baseline_holdout},
        "starting_policy": {"training": starting["training"],
                            "validation": starting_validation,
                            "holdout": starting_holdout},
        "selected": {"candidate_id": selected["candidate_id"],
                     "config": selected["config"], "training": selected["training"],
                     "validation": selected["validation"], "holdout": selected_holdout},
        "possible_overfitting": (
            selected["training"]["mean_margin"] > starting["training"]["mean_margin"]
            and selected_holdout["mean_margin"] < starting_holdout["mean_margin"]),
    }
    _write_json(output / "config_best.json", selected["config"])
    _write_json(output / "report.json", report)
    rows = []
    for candidate in candidates:
        rows.append({"candidate_id": candidate["candidate_id"],
                     "training_margin": candidate["training"]["mean_margin"],
                     "validation_margin": candidate.get("validation", {}).get("mean_margin", ""),
                     "parameters": json.dumps(candidate["config"], sort_keys=True)})
    with (output / "candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["candidate_id", "training_margin",
                                                     "validation_margin", "parameters"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path,
                        help="starting livestock policy config; defaults to the manual baseline")
    parser.add_argument("--opponent", default="starter")
    parser.add_argument("--counts", nargs="+", type=int, default=[4, 8, 12, 16])
    parser.add_argument("--species", nargs="+", choices=["COW", "SHEEP", "GOOSE"],
                        default=["COW", "SHEEP"])
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(range(1000, 1010)))
    parser.add_argument("--validation-seeds", nargs="+", type=int,
                        default=list(VALIDATION_SEEDS))
    parser.add_argument("--holdout-seeds", nargs="+", type=int, default=list(HOLDOUT_SEEDS))
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    sweep(load_config(args.base_config), tuple(args.counts), tuple(args.species),
          tuple(args.train_seeds), tuple(args.validation_seeds),
          tuple(args.holdout_seeds), args.opponent, args.output_root)


if __name__ == "__main__":
    main()
