import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from experiments.opponents import replay_action_agent, seeded_random_agent
import submission
from submission import agent

holdout_seeds = [3000, 3001, 3002, 3003, 3004]

opponents = [
    ("Starter", lambda s: "starter"),
    ("Random", lambda s: seeded_random_agent(s)),
    ("Harry Taylor ($99k)", lambda s: replay_action_agent("replays/kaggle_replays/113441225.json", 1)),
    ("GusNicho ($144k)", lambda s: replay_action_agent("replays/kaggle_replays/113076092.json", 1)),
]

print("=" * 70)
print("FINAL HOLDOUT EVALUATION OF SUBMISSION.PY (Seeds 3000-3004)")
print("=" * 70)

all_margins = []
all_rewards = []
total_wins = 0
total_games = 0

for opp_name, opp_factory in opponents:
    opp_rewards = []
    opp_margins = []
    opp_wins = 0
    for seed in holdout_seeds:
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
        opp = opp_factory(seed)
        env.run([agent, opp if isinstance(opp, str) else (lambda obs: opp(obs))])

        r0 = float(env.steps[-1][0].reward or 0)
        r1 = float(env.steps[-1][1].reward or 0)
        m = r0 - r1
        opp_rewards.append(r0)
        opp_margins.append(m)
        all_rewards.append(r0)
        all_margins.append(m)
        if m > 0:
            opp_wins += 1
            total_wins += 1
        total_games += 1

    avg_r = sum(opp_rewards) / len(opp_rewards)
    avg_m = sum(opp_margins) / len(opp_margins)
    print(f"vs {opp_name:<20}: Mean Reward=${avg_r:,.0f} | Margin=${avg_m:+,.0f} | Wins={opp_wins}/{len(holdout_seeds)}")

overall_r = sum(all_rewards) / len(all_rewards)
overall_m = sum(all_margins) / len(all_margins)
win_pct = (total_wins / total_games) * 100

print("-" * 70)
print(f"OVERALL PERFORMANCE: Mean Reward=${overall_r:,.0f} | Mean Margin=${overall_m:+,.0f} | Win Rate={win_pct:.1f}% ({total_wins}/{total_games})")
print("=" * 70)
