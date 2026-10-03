"""Live-Code League Benchmark.

Pits the submission Double Agent Champion against actual executable Python code agents:
1. Starter (Kaggle official baseline)
2. Random (Kaggle official baseline)
3. SwarmPlanterAgent (Aggressive 10-hand, multi-quadrant cash cropper)
4. DairyBaronAgent (14-animal central livestock baron)
5. CompetingAgent (Town-drain front-runner with anti-shock pricing)

All opponents make live, dynamic, turn-by-turn decisions.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make
import submission
from submission import agent as champion_agent
from experiments.opponents import seeded_random_agent
from evoexperiments.league_agents import SwarmPlanterAgent, DairyBaronAgent
import evoexperiments.competing_agent as competing_module

def make_competing_agent():
    return competing_module.CompetingAgent()

def run_tournament():
    holdout_seeds = [3000, 3001, 3002, 3003, 3004]

    opponents = [
        ("Starter (Baseline)", lambda s: "starter"),
        ("Random (Baseline)", lambda s: seeded_random_agent(s)),
        ("SwarmPlanter (Live Code)", lambda s: SwarmPlanterAgent()),
        ("DairyBaron (Live Code)", lambda s: DairyBaronAgent()),
        ("CompetingAgent (Live Code)", lambda s: make_competing_agent()),
    ]

    print("=" * 80)
    print("LIVE-CODE BENCHMARK TOURNAMENT (Seeds 3000-3004)")
    print("All opponents are interactive, decision-making Python agents.")
    print("=" * 80)

    total_wins = 0
    total_games = 0
    all_champ_rewards = []
    all_opp_rewards = []
    all_margins = []

    for opp_name, opp_factory in opponents:
        champ_scores = []
        opp_scores = []
        margins = []
        wins = 0

        for seed in holdout_seeds:
            env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
            opp = opp_factory(seed)
            
            env.run([champion_agent, opp if isinstance(opp, str) else (lambda obs: opp(obs))])

            r0 = float(env.steps[-1][0].reward or 0)
            r1 = float(env.steps[-1][1].reward or 0)
            margin = r0 - r1

            champ_scores.append(r0)
            opp_scores.append(r1)
            margins.append(margin)
            all_champ_rewards.append(r0)
            all_opp_rewards.append(r1)
            all_margins.append(margin)

            if margin > 0:
                wins += 1
                total_wins += 1
            total_games += 1

        avg_champ = sum(champ_scores) / len(champ_scores)
        avg_opp = sum(opp_scores) / len(opp_scores)
        avg_margin = sum(margins) / len(margins)
        win_rate = (wins / len(holdout_seeds)) * 100

        print(f"vs {opp_name:<28} | Win Rate: {win_rate:5.1f}% ({wins}/{len(holdout_seeds)}) | Champ: ${avg_champ:,.0f} | Opp: ${avg_opp:,.0f} | Margin: ${avg_margin:+,.0f}")

    print("-" * 80)
    overall_win_rate = (total_wins / total_games) * 100
    overall_champ_avg = sum(all_champ_rewards) / len(all_champ_rewards)
    overall_opp_avg = sum(all_opp_rewards) / len(all_opp_rewards)
    overall_margin_avg = sum(all_margins) / len(all_margins)

    print(f"OVERALL LIVE-CODE RECORD: {total_wins}/{total_games} Wins ({overall_win_rate:.1f}%)")
    print(f"Average Champion Score : ${overall_champ_avg:,.0f}")
    print(f"Average Opponent Score : ${overall_opp_avg:,.0f}")
    print(f"Average Net Margin     : ${overall_margin_avg:+,.0f}")
    print("=" * 80)

if __name__ == "__main__":
    run_tournament()
