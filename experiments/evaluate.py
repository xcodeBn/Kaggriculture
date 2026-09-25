"""Fixed-seed evaluation used by both comparisons and evolution."""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Sequence

from config import load_config
from experiments.run_games import run_game


def evaluate_config(config: dict[str, Any] | None, seeds: Sequence[int],
                    opponent: str = "random", steps: int = 720) -> dict[str, Any]:
    results = [run_game(int(seed), opponent, config or load_config(), steps)
               for seed in seeds]
    scores = [float(result["player_reward"] or 0) for result in results]
    opponent_scores = [float(result["opponent_reward"] or 0) for result in results]
    margins = [score - other for score, other in zip(scores, opponent_scores)]
    return {
        "mean_reward": statistics.fmean(scores) if scores else 0.0,
        "median_reward": statistics.median(scores) if scores else 0.0,
        "min_reward": min(scores) if scores else 0.0,
        "max_reward": max(scores) if scores else 0.0,
        "std_reward": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "mean_opponent_reward": statistics.fmean(opponent_scores) if opponent_scores else 0.0,
        "mean_margin": statistics.fmean(margins) if margins else 0.0,
        "wins": sum(margin > 0 for margin in margins),
        "losses": sum(margin < 0 for margin in margins),
        "ties": sum(margin == 0 for margin in margins),
        "games": len(scores), "seeds": list(seeds), "rewards": scores,
    }
