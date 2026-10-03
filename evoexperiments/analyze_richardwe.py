import json
from collections import Counter

with open("evoexperiments/115702423.json") as f:
    d = json.load(f)

steps = d['steps']

# Player 0 is Richardwe ($105,902)
land_buys = []
animal_buys = []
hires = Counter()
sales = Counter()
planted = Counter()

for step in range(1, len(steps)):
    day = step // 24
    hour = step % 24
    act = steps[step][0].get('action') or {}
    
    farmer = act.get('farmer', [])
    if farmer and farmer[0] == 'PLANT':
        planted[farmer[1]] += 1
    for h in act.get('hands', []):
        if h and h[0] == 'PLANT':
            planted[h[1]] += 1

    for o in act.get('market', []):
        if not o: continue
        verb = o[0]
        if verb == 'BUY_LAND':
            land_buys.append((day, hour, step))
        elif verb == 'BUY_ANIMAL':
            animal_buys.append((day, hour, o[1], o[2] if len(o)>2 else 1))
        elif verb == 'HIRE':
            hires[day] += 1
        elif verb == 'SELL':
            sales[o[1]] += o[2] if len(o)>2 else 1

print("--- Richardwe ($105,902) Land Purchases ---")
for day, hour, step in land_buys:
    print(f"  Day {day}, Hour {hour} (Step {step})")

print(f"\n--- Richardwe Animal Purchases (Total: {len(animal_buys)}) ---")
for day, hour, species, qty in animal_buys:
    print(f"  Day {day}, Hour {hour}: {species} x{qty}")

print("\n--- Richardwe Daily Hires ---")
for day in sorted(hires.keys()):
    print(f"  Day {day:2d}: {hires[day]} hands")

print("\n--- Richardwe Total Sales ---")
for prod, cnt in sales.most_common():
    print(f"  {prod}: {cnt} units")

print("\n--- Richardwe Planted Crops ---")
for crop, cnt in planted.most_common():
    print(f"  {crop}: {cnt} units")

# Now let's inspect the layout of tiles on Richardwe's farm at mid-game (Day 15) and end-game (Day 29)
def print_farm_layout(step_num, title):
    print(f"\n--- Farm Layout: {title} (Step {step_num}) ---")
    farm = steps[step_num][0]['observation']['farms'][0]
    tiles = farm['tiles']
    unlocked = farm['unlocked_quadrants']
    print(f"Unlocked Quadrants: {unlocked}")
    for y in range(len(tiles)):
        row_str = []
        for x in range(len(tiles[y])):
            t = tiles[y][x]
            if t == "LOCKED":
                row_str.append(" . ")
            elif t is None:
                row_str.append(" _ ")
            elif isinstance(t, dict):
                k = t.get("kind")
                if k == "PASTURE":
                    anim = t.get("animal")
                    row_str.append(f" {anim[0] if anim else 'P'} ")
                elif k == "COOP":
                    row_str.append(" C ")
                elif k == "PLANT":
                    crop = t.get("crop")
                    row_str.append(f" {crop[0].lower()} ")
                elif k == "WEED":
                    row_str.append(" W ")
                else:
                    row_str.append(" ? ")
            else:
                row_str.append(" ? ")
        print(f"y={y:2d} |" + "".join(row_str) + "|")

print_farm_layout(15*24, "Day 15")
print_farm_layout(29*24, "Day 29")

