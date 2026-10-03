
import json
from collections import Counter, defaultdict

with open("topmeta/116330962.json") as f:
    replay = json.load(f)

steps = replay["steps"]
print("Total steps:", len(steps))

# 1. Day 0 / Opening Analysis (Steps 0-23)
print("\n" + "=" * 80)
print("DAY 0 OPENING ANALYSIS (Steps 0 to 2)")
print("=" * 80)

for s in range(3):
    f0 = steps[s][0]
    f1 = steps[s][1]
    a0 = f0.get("action") or {}
    a1 = f1.get("action") or {}
    print(f"Step {s} P0 Action: Farmer={a0.get("farmer")} | Hands={len(a0.get("hands", []))} | Market={a0.get("market")}")
    print(f"Step {s} P1 Action: Farmer={a1.get("farmer")} | Hands={len(a1.get("hands", []))} | Market={a1.get("market")}")

# Day-by-day stats
print("\n" + "=" * 80)
print("DAY-BY-DAY PROGRESSION")
print("=" * 80)

for day in range(30):
    step_idx = day * 24
    if step_idx >= len(steps):
        break
    s = steps[step_idx]
    obs0 = s[0]["observation"]
    farms = obs0["farms"]
    
    # Analyze farm 0
    f0 = farms[0]
    m0 = f0["money"]
    h0 = len(f0["hands"])
    q0 = len(f0["unlocked_quadrants"])
    tiles0 = f0["tiles"]
    plants0 = Counter()
    animals0 = Counter()
    weeds0 = 0
    for r in tiles0:
        for t in r:
            if isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    plants0[t.get("crop")] += 1
                elif k in ("COOP", "PASTURE"):
                    if t.get("animal"):
                        animals0[t.get("animal")] += 1
                    else:
                        animals0["EMPTY_" + k] += 1
                elif k == "WEED":
                    weeds0 += 1

    # Analyze farm 1
    f1 = farms[1]
    m1 = f1["money"]
    h1 = len(f1["hands"])
    q1 = len(f1["unlocked_quadrants"])
    tiles1 = f1["tiles"]
    plants1 = Counter()
    animals1 = Counter()
    weeds1 = 0
    for r in tiles1:
        for t in r:
            if isinstance(t, dict):
                k = t.get("kind")
                if k == "PLANT":
                    plants1[t.get("crop")] += 1
                elif k in ("COOP", "PASTURE"):
                    if t.get("animal"):
                        animals1[t.get("animal")] += 1
                    else:
                        animals1["EMPTY_" + k] += 1
                elif k == "WEED":
                    weeds1 += 1

    p0_str = " ".join(f"{c}:{cnt}" for c, cnt in plants0.items())
    a0_str = " ".join(f"{a}:{cnt}" for a, cnt in animals0.items())
    p1_str = " ".join(f"{c}:{cnt}" for c, cnt in plants1.items())
    a1_str = " ".join(f"{a}:{cnt}" for a, cnt in animals1.items())
    
    print(f"Day {day:2d} | P0: ${m0:6.0f} Q={q0} H={h0:2d} W={weeds0:2d} | Anim=[{a0_str}] | Crops=[{p0_str}]")
    print(f"       | P1: ${m1:6.0f} Q={q1} H={h1:2d} W={weeds1:2d} | Anim=[{a1_str}] | Crops=[{p1_str}]")
