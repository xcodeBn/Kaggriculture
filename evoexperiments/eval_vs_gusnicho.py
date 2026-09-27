import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission import COMPETITION_CONFIG
from evoexperiments.parallel_eval import parallel_evaluate_pool

def main():
    holdout_seeds = (3000, 3001, 3002, 3003, 3004)
    opp = "replay:replays/kaggle_replays/113076092.json:1"

    print("Evaluating Current Submission vs GusNicho (113076092.json:1)...")
    res_curr = parallel_evaluate_pool(COMPETITION_CONFIG, holdout_seeds, [opp], steps=720)
    print(f"Current Submission vs GusNicho: Mean Margin = ${int(res_curr['mean_margin']):,} | Wins = {res_curr['wins']}/{res_curr['games']}")

    print("\nEvaluating Enhanced Agent vs GusNicho (113076092.json:1)...")
    res_enh = parallel_evaluate_pool("evoexperiments/enhanced_agent.py", holdout_seeds, [opp], steps=720)
    print(f"Enhanced Agent vs GusNicho: Mean Margin = ${int(res_enh['mean_margin']):,} | Wins = {res_enh['wins']}/{res_enh['games']}")

    diff = int(res_enh['mean_margin'] - res_curr['mean_margin'])
    print(f"\nNet Difference vs GusNicho: ${diff:+,}")

if __name__ == "__main__":
    main()
