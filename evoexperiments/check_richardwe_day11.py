import json

with open("evoexperiments/115702423.json") as f:
    d = json.load(f)

steps = d['steps']

for s_idx in range(11*24, 13*24):
    day = s_idx // 24
    hour = s_idx % 24
    act = steps[s_idx][0].get('action') or {}
    orders = act.get('market', [])
    obs = steps[s_idx][0].get('observation')
    m = obs['farms'][0]['money']
    sales = [o for o in orders if o and o[0] == 'SELL']
    buys = [o for o in orders if o and o[0] != 'SELL']
    if sales or buys:
        print(f"Step {s_idx} (Day {day}, Hr {hour}) Money=${m:6.0f} | Sales: {sales} | Buys: {buys}")
