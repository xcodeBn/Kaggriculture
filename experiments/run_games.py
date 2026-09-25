"""Run local Kaggriculture episodes and optionally save complete replays."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_config  # noqa: E402
import submission  # noqa: E402
from experiments.opponents import seeded_random_agent  # noqa: E402


def load_environment():
    try:
        from kaggle_environments import make
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt to run the local simulator") from exc
    return make


def run_game(seed: int, opponent: str = "random", config: dict[str, Any] | None = None,
             steps: int = 720, save_dir: Path | None = None) -> dict[str, Any]:
    make = load_environment()
    submission.set_policy_config(config or load_config())
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=True)
    opponent_agent = seeded_random_agent(seed) if opponent == "random" else opponent
    env.run([submission.agent, opponent_agent])
    final = env.steps[-1]
    rewards = [float(state.reward or 0) for state in final]
    statuses = [state.status for state in final]
    observations = [getattr(state, "observation", {}) or {} for state in final]
    observed_farms = observations[0].get("farms", []) if observations else []
    final_money = [farm.get("money") for farm in observed_farms]
    summary: dict[str, Any] = {
        "seed": seed, "opponent": opponent, "steps": len(env.steps),
        "rewards": rewards, "player_reward": rewards[0] if rewards else None,
        "opponent_reward": rewards[1] if len(rewards) > 1 else None,
        "final_money": final_money,
        "player_money": final_money[0] if final_money else None,
        "opponent_money": final_money[1] if len(final_money) > 1 else None,
        "statuses": statuses,
    }
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = f"game_seed_{seed}"
        replay_path = save_dir / f"{stem}.json"
        with replay_path.open("x", encoding="utf-8") as stream:
            json.dump(env.toJSON(), stream, separators=(",", ":"))
        summary["replay"] = str(replay_path)
        with (save_dir / f"{stem}.summary.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
    return summary


def seed_list(seeds: Iterable[int] | None, start_seed: int | None, num_games: int | None) -> list[int]:
    if seeds:
        return list(seeds)
    if start_seed is not None and num_games is not None:
        return list(range(start_seed, start_seed + num_games))
    raise ValueError("Pass --seeds or both --start-seed and --num-games")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--start-seed", type=int)
    parser.add_argument("--num-games", type=int)
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "replays")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_dir = args.output_dir / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    for seed in seed_list(args.seeds, args.start_seed, args.num_games):
        result = run_game(seed, args.opponent, cfg, args.steps, run_dir)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
