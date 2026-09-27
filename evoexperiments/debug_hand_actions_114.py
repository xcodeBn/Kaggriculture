import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from config import load_config
from submission import COMPETITION_CONFIG, set_policy_config, agent, animal_goals, ANIMAL_INFO, distance

mixed_config = load_config(ROOT / "optimized_config.json")
mixed_config.update({
    "buy_land_enabled": True,
    "land_reserve": 250,
    "buy_land_cash_multiplier": 1.15,
    "cash_reserve": 300,
    "max_animals": 12,
    "cow_goal": 6,
    "sheep_goal": 6,
    "max_hands": 8,
    "min_cash_for_hand": 50,
    "sell_batch_size": 10,
    "sell_price_ratio": 0.50,
})

def debug_step(obs):
    set_policy_config(mixed_config)
    step = obs.get("step", 0)
    if step == 114:
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs["private"]
        shed = private["shed"]
        goals = animal_goals(mixed_config)
        hands = farm.get("hands", []) or []
        inventories = private.get("inventories", []) or []
        hand_inventories = [inventories[i + 1] if i + 1 < len(inventories) else {}
                            for i in range(len(hands))]
        positions = [tuple(position) for position in hands]
        
        board_size = len(farm.get("tiles", []))
        half = board_size // 2
        shed_positions = [(half - 1, half - 1), (half, half - 1),
                          (half - 1, half), (half, half)]
        
        empty_structures = {}
        animals = []
        jobs = []
        for y, row in enumerate(farm.get("tiles", [])):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue
                species = tile.get("animal")
                if species in goals:
                    animals.append((x, y, tile))
                    if not tile.get("fed_today", False):
                        jobs.append({"priority": 0, "target": (x, y),
                                     "action": ["FEED"], "needs": "WHEAT"})
                    if tile.get("yield_units", 0) > 0:
                        jobs.append({"priority": 1, "target": (x, y), "action": ["HARVEST"]})
                    if tile.get("fertilizer_available", False):
                        jobs.append({"priority": 2, "target": (x, y),
                                     "action": ["COLLECT_FERTILIZER"]})
                    if not tile.get("cared_today", False):
                        jobs.append({"priority": 3, "target": (x, y), "action": ["CARE"]})
                elif tile.get("kind") in {ANIMAL_INFO[s]["structure"] for s in goals} and not tile.get("animal"):
                    empty_structures.setdefault(tile["kind"], []).append((x, y))

        carried_animals = sum(int(inv.get(species, 0)) for inv in inventories
                              for species in goals)
        available_places = sum(len(tiles) for tiles in empty_structures.values())
        stock_left_to_pick = max(0, available_places - carried_animals)
        place_slots = [position for structure in empty_structures.values() for position in structure]
        for species in goals:
            stock = int(shed.get(species, 0))
            animals_to_pick = min(stock, stock_left_to_pick)
            for index in range(animals_to_pick):
                jobs.append({"priority": 4, "target": shed_positions[index % len(shed_positions)],
                             "action": ["PICKUP", species, 1]})
            stock_left_to_pick -= animals_to_pick

        print("Total jobs generated at 114:", len(jobs))
        for j in jobs:
            print("  Job:", j)
            
        print("\nHands and matching at 114:")
        for idx, (pos, inv) in enumerate(zip(positions, hand_inventories)):
            eligible = [job for job in jobs if not job.get("needs") or inv.get(job["needs"], 0) > 0]
            print(f"Hand {idx} at {pos}, inv={inv}, eligible_count={len(eligible)}")
            if eligible:
                best = min(eligible, key=lambda item: (item["priority"], distance(pos, item["target"])))
                print(f"   -> Chose {best}")
                jobs.remove(best)
            else:
                print("   -> NO ELIGIBLE JOB")
                
    return agent(obs)

env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 3000}, debug=True)
env.run([debug_step, "starter"])
