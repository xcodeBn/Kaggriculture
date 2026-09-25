"""Compare baseline and candidate on identical, fixed seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import TRAIN_SEEDS, load_config
from experiments.evaluate import evaluate_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAIN_SEEDS))
    parser.add_argument("--steps", type=int, default=720)
    args = parser.parse_args()
    baseline = evaluate_config(load_config(), args.seeds, steps=args.steps)
    candidate = evaluate_config(load_config(args.candidate), args.seeds, steps=args.steps)
    print(json.dumps({"seeds": args.seeds, "baseline": baseline, "candidate": candidate,
                      "mean_reward_delta": candidate["mean_reward"] - baseline["mean_reward"]}, indent=2))


if __name__ == "__main__":
    main()
