"""Multi-core parallel game evaluator for Kaggriculture.

Accelerates episode evaluations across multiple CPU cores using ProcessPoolExecutor.
Provides identical statistical outputs and random seeding as evaluate.py.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config
from experiments.run_games import run_game


import contextlib
import io

def _worker_run_game(args: tuple[int, str, Any, int]) -> dict[str, Any]:
    seed, opponent, agent_or_config, steps = args
    if isinstance(agent_or_config, str) and agent_or_config.endswith(".py"):
        from evoexperiments.env_loader import get_clean_make
        make = get_clean_make()
        from experiments.opponents import replay_action_agent, seeded_random_agent
        import importlib.util
        mod_path = Path(agent_or_config)
        if not mod_path.is_absolute():
            mod_path = ROOT / mod_path
        spec = importlib.util.spec_from_file_location("custom_eval_agent", str(mod_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        agent_fn = mod.agent

        env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
        if opponent == "random":
            opp_fn = seeded_random_agent(seed)
        elif opponent.startswith("replay:"):
            replay_spec = opponent[len("replay:"):]
            try:
                replay_path, player_text = replay_spec.rsplit(":", 1)
                replay_player = int(player_text)
            except ValueError:
                replay_path, replay_player = replay_spec, 1
            opp_fn = replay_action_agent(replay_path, replay_player)
        else:
            opp_fn = opponent

        env.run([agent_fn, opp_fn])
        final = env.steps[-1]
        rewards = [float(state.reward or 0) for state in final]
        statuses = [state.status for state in final]
        obs_farms = getattr(final[0], "observation", {}).get("farms", [])
        final_money = [farm.get("money") for farm in obs_farms]
        return {
            "seed": seed, "opponent": opponent, "steps": len(env.steps),
            "rewards": rewards, "player_reward": rewards[0] if rewards else None,
            "opponent_reward": rewards[1] if len(rewards) > 1 else None,
            "final_money": final_money,
            "player_money": final_money[0] if final_money else None,
            "opponent_money": final_money[1] if len(final_money) > 1 else None,
            "statuses": statuses,
        }
    else:
        return run_game(seed, opponent, agent_or_config, steps)


def parallel_evaluate_config(
    config: dict[str, Any] | None,
    seeds: Sequence[int],
    opponent: str = "random",
    steps: int = 720,
    max_workers: int | None = None
) -> dict[str, Any]:
    """Run seeds in parallel across CPU cores."""
    effective_config = config or load_config()
    workers = max_workers or max(1, min(os.cpu_count() or 1, len(seeds)))
    
    tasks = [(int(s), opponent, effective_config, steps) for s in seeds]
    
    if workers > 1 and len(seeds) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_worker_run_game, tasks))
    else:
        results = [_worker_run_game(t) for t in tasks]

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
        "games": len(scores),
        "seeds": list(seeds),
        "rewards": scores,
        "opponent_rewards": opponent_scores,
    }


def parallel_evaluate_pool(
    config: dict[str, Any] | None,
    seeds: Sequence[int],
    opponents: Sequence[str],
    steps: int = 720,
    max_workers: int | None = None
) -> dict[str, Any]:
    """Evaluate against multiple opponents in parallel."""
    effective_config = config if config is not None else load_config()
    workers = max_workers or max(1, os.cpu_count() or 1)
    
    # Create all (seed, opponent) tasks
    tasks = []
    task_keys = []
    for opp in opponents:
        for s in seeds:
            tasks.append((int(s), opp, effective_config, steps))
            task_keys.append((opp, int(s)))
            
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            raw_results = list(executor.map(_worker_run_game, tasks))
    else:
        raw_results = [_worker_run_game(t) for t in tasks]

    # Regroup results by opponent
    by_opponent_data: dict[str, list[dict[str, Any]]] = {opp: [] for opp in opponents}
    for (opp, s), res in zip(task_keys, raw_results):
        by_opponent_data[opp].append(res)

    by_opponent = {}
    for opp, res_list in by_opponent_data.items():
        scores = [float(r["player_reward"] or 0) for r in res_list]
        opp_scores = [float(r["opponent_reward"] or 0) for r in res_list]
        margins = [s - o for s, o in zip(scores, opp_scores)]
        by_opponent[opp] = {
            "mean_reward": statistics.fmean(scores) if scores else 0.0,
            "mean_opponent_reward": statistics.fmean(opp_scores) if opp_scores else 0.0,
            "mean_margin": statistics.fmean(margins) if margins else 0.0,
            "wins": sum(m > 0 for m in margins),
            "losses": sum(m < 0 for m in margins),
            "ties": sum(m == 0 for m in margins),
            "games": len(scores),
            "rewards": scores,
            "opponent_rewards": opp_scores,
        }

    all_margins = [res["mean_margin"] for res in by_opponent.values()]
    all_rewards = [r for res in by_opponent.values() for r in res["rewards"]]
    all_opp_rewards = [r for res in by_opponent.values() for r in res["opponent_rewards"]]
    pooled_margins = [s - o for s, o in zip(all_rewards, all_opp_rewards)]

    return {
        "mean_reward": statistics.fmean(all_rewards) if all_rewards else 0.0,
        "mean_opponent_reward": statistics.fmean(all_opp_rewards) if all_opp_rewards else 0.0,
        "mean_margin": statistics.fmean(all_margins) if all_margins else 0.0,
        "worst_opponent_mean_margin": min(all_margins) if all_margins else 0.0,
        "pooled_std_margin": statistics.pstdev(pooled_margins) if len(pooled_margins) > 1 else 0.0,
        "wins": sum(r["wins"] for r in by_opponent.values()),
        "losses": sum(r["losses"] for r in by_opponent.values()),
        "ties": sum(r["ties"] for r in by_opponent.values()),
        "games": sum(r["games"] for r in by_opponent.values()),
        "opponents": list(opponents),
        "by_opponent": by_opponent,
    }


if __name__ == "__main__":
    t0 = time.time()
    print("Testing parallel evaluation smoke test on 4 seeds (steps=48)...")
    res = parallel_evaluate_config(None, seeds=[1000, 1001, 1002, 1003], opponent="random", steps=48)
    dt = time.time() - t0
    print(f"Finished 4 games in {dt:.2f}s! Mean reward: {res['mean_reward']}, Margin: {res['mean_margin']}")
