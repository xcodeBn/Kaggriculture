"""Evaluate updated submission.py on untouched holdout seeds (3000-3004)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission import COMPETITION_CONFIG
from evoexperiments.parallel_eval import parallel_evaluate_pool

def main():
    holdout_seeds = (3000, 3001, 3002, 3003, 3004)
    opponents = ("random", "starter", "replay:replays/kaggle_replays/113441225.json:1")

    print("==================================================================")
    print("VERIFYING UPDATED SUBMISSION.PY (Holdout Seeds 3000-3004)...")
    print("==================================================================")
    res = parallel_evaluate_pool(COMPETITION_CONFIG, holdout_seeds, opponents, steps=720)
    print(f"Updated Submission.py: Mean Margin = ${int(res['mean_margin']):,} | Wins = {res['wins']}/{res['games']}")
    for opp in opponents:
        opp_margin = int(res['by_opponent'][opp]['mean_margin'])
        opp_wins = res['by_opponent'][opp]['wins']
        print(f"   vs {opp}: ${opp_margin:,} (Wins: {opp_wins}/5)")

    target = 22340
    delta = int(res['mean_margin'] - target)
    print("==================================================================")
    print(f"NET IMPROVEMENT OVER ORIGINAL SUBMISSION ($22,340): ${delta:+,}")
    print("==================================================================")

if __name__ == "__main__":
    main()
