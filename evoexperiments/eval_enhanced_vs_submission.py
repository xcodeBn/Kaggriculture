"""Compare EnhancedAgent vs Current Submission on Holdout Seeds (3000-3004)."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission import COMPETITION_CONFIG
import evoexperiments.enhanced_agent as enhanced_agent
from evoexperiments.parallel_eval import parallel_evaluate_pool

def main():
    holdout_seeds = (3000, 3001, 3002, 3003, 3004)
    opponents = ("random", "starter", "replay:replays/kaggle_replays/113441225.json:1")

    print("==================================================================")
    print("1. EVALUATING CURRENT SUBMISSION.PY (Holdout Seeds 3000-3004)...")
    print("==================================================================")
    res_curr = parallel_evaluate_pool(COMPETITION_CONFIG, holdout_seeds, opponents, steps=720)
    print(f"Current Submission: Mean Margin = ${int(res_curr['mean_margin']):,} | Wins = {res_curr['wins']}/{res_curr['games']}")
    for opp in opponents:
        opp_margin = int(res_curr['by_opponent'][opp]['mean_margin'])
        opp_wins = res_curr['by_opponent'][opp]['wins']
        print(f"   vs {opp}: ${opp_margin:,} (Wins: {opp_wins}/5)")

    print("\n==================================================================")
    print("2. EVALUATING ENHANCED AGENT (Holdout Seeds 3000-3004)...")
    print("==================================================================")
    res_enh = parallel_evaluate_pool("evoexperiments/enhanced_agent.py", holdout_seeds, opponents, steps=720)
    print(f"Enhanced Agent: Mean Margin = ${int(res_enh['mean_margin']):,} | Wins = {res_enh['wins']}/{res_enh['games']}")
    for opp in opponents:
        opp_margin = int(res_enh['by_opponent'][opp]['mean_margin'])
        opp_wins = res_enh['by_opponent'][opp]['wins']
        print(f"   vs {opp}: ${opp_margin:,} (Wins: {opp_wins}/5)")

    delta = int(res_enh['mean_margin'] - res_curr['mean_margin'])
    print("\n==================================================================")
    print(f"NET IMPROVEMENT ON HOLDOUT SEEDS: ${delta:+,}")
    print("==================================================================")

if __name__ == "__main__":
    main()
