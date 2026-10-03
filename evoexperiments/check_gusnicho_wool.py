import json

with open("replays/kaggle_replays/113076092.json") as f:
    d = json.load(f)

for step in range(0, 720, 24):
    frame = d['steps'][step]
    obs = frame[0]['observation']
    p_wool = obs['market']['prices']['WOOL']
    inv_wool = obs['market']['inventory']['WOOL']
    shops = obs['town']['unlocked_shops']
    # Check GusNicho action at step
    act = frame[1].get('action') or {}
    sells = [o for o in act.get('market', []) if o and o[0] == 'SELL' and o[1] == 'WOOL']
    print(f"Step {step:3d} (Day {step//24:2d}): Wool Price=${p_wool:3d} (Inv={inv_wool}) | Wool sells={sells} | Shops={shops}")
