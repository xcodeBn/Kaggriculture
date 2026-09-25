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
        "opponent_rewards": opponent_scores,
    }


def evaluate_opponent_pool(config: dict[str, Any] | None, seeds: Sequence[int],
                           opponents: Sequence[str], steps: int = 720) -> dict[str, Any]:
    """Evaluate identical seeds against each opponent and average opponents equally."""
    if not opponents:
        raise ValueError("At least one opponent is required")
    if len(set(opponents)) != len(opponents):
        raise ValueError("Opponent names must be unique")
    by_opponent = {
        name: evaluate_config(config, seeds, opponent=name, steps=steps)
        for name in opponents
    }
    margins = [result["mean_margin"] for result in by_opponent.values()]
    rewards = [reward for result in by_opponent.values() for reward in result["rewards"]]
    opponent_rewards = [reward for result in by_opponent.values()
                        for reward in result["opponent_rewards"]]
    pooled_margins = [margin for result in by_opponent.values()
                      for margin in (player - rival for player, rival in
                                     zip(result["rewards"], result["opponent_rewards"]))]
    return {
        "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
        "median_reward": statistics.median(rewards) if rewards else 0.0,
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_opponent_reward": statistics.fmean(opponent_rewards) if opponent_rewards else 0.0,
        "mean_margin": statistics.fmean(margins) if margins else 0.0,
        "worst_opponent_mean_margin": min(margins) if margins else 0.0,
        "pooled_std_margin": statistics.pstdev(pooled_margins) if len(pooled_margins) > 1 else 0.0,
        "wins": sum(result["wins"] for result in by_opponent.values()),
        "losses": sum(result["losses"] for result in by_opponent.values()),
        "ties": sum(result["ties"] for result in by_opponent.values()),
        "games": sum(result["games"] for result in by_opponent.values()),
        "seeds": list(seeds), "opponents": list(opponents),
        "by_opponent": by_opponent,
    }
