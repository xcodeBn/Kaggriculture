"""Inspect GusNicho's exact timeline in 113076092.json to reverse-engineer $144k strategy."""
import json
from pathlib import Path
from collections import Counter

rpath = Path("replays/kaggle_replays/113076092.json")
with open(rpath, "r", encoding="utf-8") as f:
    data = json.load(f)

steps = data.get("steps", [])
print(f"Total steps: {len(steps)}")

# Player 1 is GusNicho
land_buys = []
animal_buys = []
hires = Counter()
sales = Counter()
wheat_buys = []

for f_idx in range(1, len(steps)):
    act = steps[f_idx][1].get("action") or {}
    obs = steps[f_idx][0].get("observation") or {} # step f_idx
    day = f_idx // 24
    hour = f_idx % 24
    
    market_orders = act.get("market") or []
    for o in market_orders:
        if not isinstance(o, (list, tuple)) or not o:
            continue
        verb = o[0]
        if verb == "BUY_LAND":
            land_buys.append((day, hour, f_idx))
        elif verb == "BUY_ANIMAL":
            animal_buys.append((day, hour, o[1]))
        elif verb == "HIRE":
            hires[day] += 1
        elif verb == "BUY_PRODUCT" and len(o) > 1 and o[1] == "WHEAT":
            wheat_buys.append((day, hour, o[2] if len(o) > 2 else 1))
        elif verb == "SELL":
            sales[o[1]] += o[2] if len(o) > 2 else 1

print("\n--- GusNicho Land Purchases ---")
for day, hour, step in land_buys:
    print(f"  Day {day} (Hour {hour}, Step {step})")

print(f"\n--- GusNicho Animal Purchases (Total: {len(animal_buys)}) ---")
for day, hour, sp in animal_buys:
    print(f"  Day {day} (Hour {hour}): {sp}")

print("\n--- GusNicho Daily Hires (Sample days) ---")
for d in sorted(hires.keys())[:10]:
    print(f"  Day {d}: {hires[d]} hands hired")
print(f"  Day 20: {hires.get(20, 0)} hands")
print(f"  Day 28: {hires.get(28, 0)} hands")

print("\n--- GusNicho Total Units Sold ---")
for prod, count in sales.most_common():
    print(f"  {prod}: {count} units")
