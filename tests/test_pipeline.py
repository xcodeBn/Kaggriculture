from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DEFAULT_CONFIG, PARAMETERS, load_config
from optimizer.evolve import crossover, mutate
import submission
from analysis.analyze_replay import analyze_replay
from experiments.opponents import seeded_random_agent


class PolicyTests(unittest.TestCase):
    def setUp(self):
        submission.set_policy_config(DEFAULT_CONFIG)

    def test_config_loading(self):
        cfg = load_config()
        self.assertEqual(cfg["cash_reserve"], 500)
        self.assertEqual(cfg, DEFAULT_CONFIG)
        self.assertEqual(set(cfg), set(DEFAULT_CONFIG))
        with tempfile.TemporaryDirectory() as directory:
            bad_path = Path(directory) / "bad.json"
            bad_path.write_text(json.dumps({"max_hands": 99}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(bad_path)

    def test_repeated_shop_demand_counts_instances(self):
        obs = {"town": {"unlocked_shops": ["BAKERY", "BAKERY", "PET_CAFE"]}}
        demand = submission.town_product_demand(obs)
        self.assertEqual(demand["WHEAT"], 2)
        self.assertEqual(demand["EGG"], 2)
        self.assertEqual(demand["CARROT"], 2)

    def test_crop_selection_uses_config_ratios(self):
        obs = {"market": {"prices": {crop: submission.CROP_INFO[crop]["base_price"]
                                      for crop in submission.CROP_ORDER}},
               "town": {"unlocked_shops": []}}
        default_crop = submission.choose_crop(obs)
        cfg = dict(DEFAULT_CONFIG, melon_ratio=2.0, wheat_ratio=0.25,
                   carrot_ratio=0.25, tomato_ratio=0.25, strawberry_ratio=0.25)
        submission.set_policy_config(cfg)
        self.assertEqual(submission.choose_crop(obs), "MELON")
        self.assertTrue(default_crop in submission.CROP_ORDER)

    def test_market_selling_threshold(self):
        obs = {"private": {"shed": {"WHEAT": 3}},
               "market": {"prices": {"WHEAT": 10}}}
        self.assertEqual(submission.sell_orders(obs), [])
        obs["market"]["prices"]["WHEAT"] = 25
        self.assertEqual(submission.sell_orders(obs), [["SELL", "WHEAT", 3]])

    def test_crop_not_harvested_before_documented_first_yield(self):
        farm = {"tiles": [[{"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
                            "yield_units": 1}]]}
        self.assertIsNone(submission.find_harvest_target(farm, 0, (0, 0)))
        self.assertEqual(submission.find_harvest_target(farm, 2, (0, 0)), (0, 0))

    def test_seed_purchase_respects_reserve(self):
        obs = {"player": 0, "farms": [{"money": 500, "unlocked_quadrants": ["NW"],
                                        "tiles": [[None] * 5], "farmer": [0, 0], "hands": [],
                                        "hires_today": 0}],
               "private": {"seeds": {}, "shed": {}},
               "market": {"prices": {}, "inventory": {}}, "town": {"unlocked_shops": []}}
        orders = submission.KaggricultureAgent(DEFAULT_CONFIG)(obs)["market"]
        self.assertFalse(any(order[0] == "BUY_SEED" for order in orders))
        buy_cfg = dict(DEFAULT_CONFIG, cash_reserve=0, min_cash_for_hand=9999)
        obs["farms"][0]["money"] = 200
        orders = submission.KaggricultureAgent(buy_cfg)(obs)["market"]
        self.assertTrue(any(order[0] == "BUY_SEED" for order in orders))

    def test_action_format(self):
        obs = {"player": 0, "farms": [{"money": 3000, "tiles": [[None]],
                                        "farmer": [0, 0], "hands": [], "unlocked_quadrants": ["NW"],
                                        "hires_today": 0}],
               "private": {"seeds": {}, "shed": {}}, "market": {}, "town": {}}
        result = submission.agent(obs)
        self.assertEqual(set(result), {"farmer", "hands", "market"})
        self.assertIsInstance(result["farmer"], list)
        self.assertIsInstance(result["hands"], list)
        self.assertIsInstance(result["market"], list)

    def test_mutation_and_crossover_are_bounded(self):
        rng = random.Random(9)
        mutated = mutate(DEFAULT_CONFIG, rng, rate=1.0)
        mixed = crossover(DEFAULT_CONFIG, mutated, rng)
        self.assertEqual(set(mutated), set(DEFAULT_CONFIG))
        self.assertTrue(100 <= mutated["cash_reserve"] <= 1500)
        self.assertTrue(0.5 <= mixed["sell_price_ratio"] <= 1.0)
        self.assertEqual(mutate(DEFAULT_CONFIG, random.Random(9), 1.0), mutated)
        PARAMETERS["test_category"] = {"type": "categorical", "choices": ["a", "b"]}
        try:
            categorical_parent = dict(DEFAULT_CONFIG, test_category="a")
            categorical_child = mutate(categorical_parent, random.Random(2), 1.0)
            categorical_cross = crossover(categorical_parent, categorical_child, random.Random(3))
            self.assertIn(categorical_child["test_category"], {"a", "b"})
            self.assertIn(categorical_cross["test_category"], {"a", "b"})
        finally:
            PARAMETERS.pop("test_category")

    def test_replay_parsing(self):
        replay = {"steps": [
            [
                {"action": {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]},
                 "observation": {"farms": [{"money": 100, "farmer": [0, 0], "hands": [],
                                              "tiles": [[None]]}, {"money": 90}],
                                 "town": {"unlocked_shops": ["BAKERY", "BAKERY"]}}, "reward": None},
                {"action": {}, "observation": {}, "reward": None},
            ],
            [
                {"action": {"farmer": ["HARVEST"], "hands": [], "market": [["SELL", "WHEAT", 2]]},
                 "observation": {"farms": [{"money": 120, "farmer": [0, 0], "hands": [[0, 0]],
                                              "tiles": [[{"kind": "PLANT", "crop": "WHEAT"}]]},
                                            {"money": 100}],
                                 "town": {"unlocked_shops": ["BAKERY", "BAKERY"]}}, "reward": 1},
                {"action": {}, "observation": {}, "reward": -1},
            ],
            [
                {"action": {"farmer": ["HARVEST"], "hands": [["HARVEST"]], "market": []},
                 "observation": {"farms": [{"money": 130, "farmer": [0, 0], "hands": [[0, 0]],
                                              "tiles": [[None]]}, {"money": 100}],
                                 "town": {"unlocked_shops": ["BAKERY", "BAKERY"]}}, "reward": 1},
                {"action": {}, "observation": {}, "reward": -1},
            ],
        ]}
        result = analyze_replay(replay)
        self.assertEqual(result["outcome"]["result"], "win")
        self.assertEqual(result["crops"]["plant_actions_attempted_by_crop"], {"WHEAT": 1})
        self.assertEqual(result["crops"]["plant_transitions_observed_by_crop"], {"WHEAT": 1})
        self.assertEqual(result["crops"]["harvest_actions_attempted_by_crop"], {"WHEAT": 3})
        self.assertEqual(result["crops"]["harvests_confirmed_by_state_change"], {"WHEAT": 1})
        self.assertEqual(result["town"]["shop_instances_observed"], {"BAKERY": 2})
        self.assertEqual(result["economy"]["sale_order_units_requested"], {"WHEAT": 2})

    def test_seeded_random_opponent_repeats_for_same_turn(self):
        obs = {"player": 1, "step": 12, "farms": [{}, {"money": 3000, "hands": []}],
               "private": {"seeds": {"WHEAT": 0, "CARROT": 0}}}
        self.assertEqual(seeded_random_agent(42)(obs), seeded_random_agent(42)(obs))


if __name__ == "__main__":
    unittest.main()
