"""Test Land Expansion and Mixed Herd Policy against Holdout Seeds."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from config import load_config
from submission import COMPETITION_CONFIG, set_policy_config
from evoexperiments.parallel_eval import parallel_evaluate_pool

# Design Mixed Herd + Land Expansion candidate
mixed_config = load_config(ROOT / "optimized_config.json")
mixed_config.update({
    "buy_land_enabled": True,
    "land_reserve": 250,
    "buy_land_cash_multiplier": 1.15,
    "cash_reserve": 300,
    "max_animals": 12,
    "cow_goal": 6,
    "sheep_goal": 6,
    "max_hands": 8,
    "min_cash_for_hand": 50,
    "sell_batch_size": 10,
    "sell_price_ratio": 0.50,
})

def main():
    holdout_seeds = (3000, 3001, 3002, 3003, 3004)
    opponents = ("random", "starter", "replay:replays/kaggle_replays/113441225.json:1")

    print("Evaluating Current Submission (Capped 8 Cows, 0 Land)...")
    res_curr = parallel_evaluate_pool(COMPETITION_CONFIG, holdout_seeds, opponents, steps=720)
    print(f"Current Submission: Mean Margin = ${int(res_curr['mean_margin']):,} | Wins = {res_curr['wins']}/{res_curr['games']}")

    print("\nEvaluating Mixed Herd + Land Expansion (6 Cows + 6 Sheep + Land Buy)...")
    res_mixed = parallel_evaluate_pool(mixed_config, holdout_seeds, opponents, steps=720)
    print(f"Mixed Herd + Land: Mean Margin = ${int(res_mixed['mean_margin']):,} | Wins = {res_mixed['wins']}/{res_mixed['games']}")

    delta = int(res_mixed['mean_margin'] - res_curr['mean_margin'])
    print(f"\nNet Improvement over Current Submission: ${delta:+,}")

    print("\nBreakdown by Opponent:")
    for opp in opponents:
        curr_m = int(res_curr['by_opponent'][opp]['mean_margin'])
        new_m = int(res_mixed['by_opponent'][opp]['mean_margin'])
        print(f"  vs {opp}: Current ${curr_m:,} -> New ${new_m:,} (Diff: ${new_m - curr_m:+,})")


if __name__ == "__main__":
    main()
