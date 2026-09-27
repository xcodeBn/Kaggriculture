import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evoexperiments.league_agents import DairyBaronAgent, SwarmPlanterAgent
from kaggle_environments import make
from experiments.opponents import replay_action_agent, seeded_random_agent

def evaluate():
    with open('evoexperiments/checkpoints/best_league_champion.json') as f:
        champ_data = json.load(f)

    dairy_cfg = champ_data['dairy_baron']
    swarm_cfg = champ_data['swarm_planter']

    opponents = [
        ('random', lambda s: seeded_random_agent(s)),
        ('starter', lambda s: 'starter'),
        ('harry_taylor', lambda s: replay_action_agent('replays/kaggle_replays/113441225.json', 1))
    ]

    for name, agent_cls, cfg in [('DairyBaron', DairyBaronAgent, dairy_cfg), ('SwarmPlanter', SwarmPlanterAgent, swarm_cfg)]:
        print(f"--- Evaluating {name} on Holdout Seeds 3000-3004 ---")
        margins = []
        rewards = []
        wins = 0
        for opp_name, opp_factory in opponents:
            opp_margins = []
            for seed in range(3000, 3005):
                bot = agent_cls(cfg)
                opp = opp_factory(seed)
                env = make('kaggriculture', configuration={'episodeSteps': 720, 'seed': seed}, debug=False)
                env.run([lambda obs: bot(obs), opp])
                r0 = float(env.steps[-1][0].reward or 0)
                r1 = float(env.steps[-1][1].reward or 0)
                m = r0 - r1
                margins.append(m)
                rewards.append(r0)
                opp_margins.append(m)
                if m > 0:
                    wins += 1
            avg_opp_m = sum(opp_margins) / len(opp_margins)
            print(f"  vs {opp_name:12s}: Mean Margin=${avg_opp_m:+,.0f}")
        
        avg_rew = sum(rewards) / len(rewards)
        avg_mar = sum(margins) / len(margins)
        print(f"==> {name} OVERALL: Mean Reward=${avg_rew:,.0f} | Mean Margin=${avg_mar:+,.0f} | Wins={wins}/15\n")

if __name__ == '__main__':
    evaluate()
