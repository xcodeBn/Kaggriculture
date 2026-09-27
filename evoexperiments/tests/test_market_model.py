"""Test market model price calculation against official specification in README.md."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Avoid shadowing by Python's built-in evoexperiments module:
import importlib.util
spec = importlib.util.spec_from_file_location("market_model", str(ROOT / "evoexperiments" / "market_model.py"))
market_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(market_model)

MARKET_PARAMS = market_model.MARKET_PARAMS
calculate_price = market_model.calculate_price
compute_expected_proceeds = market_model.compute_expected_proceeds
estimate_inventory_from_price = market_model.estimate_inventory_from_price
optimal_sale_batch = market_model.optimal_sale_batch


class TestMarketModel(unittest.TestCase):
    def test_official_price_table(self):
        """Verify exact benchmark prices from README.md table."""
        test_cases = [
            # Resource: (Base, P(I0-T), P(I0+T), P(I0+2T))
            ("WHEAT", 25, 45, 20, 19),
            ("CARROT", 35, 70, 10, 1),
            ("TOMATO", 60, 84, 24, 9),
            ("STRAWBERRY", 120, 204, 1, 1),
            ("MELON", 250, 300, 1, 1),
            ("EGG", 50, 70, 40, 39),
            ("MILK", 160, 256, 1, 1),
            ("WOOL", 200, 240, 1, 1),
            ("FERTILIZER", 100, 140, 60, 20),
        ]

        for item, base, p_below_T, p_above_T, p_above_2T in test_cases:
            params = MARKET_PARAMS[item]
            I0 = params["I0"]
            T = params["T"]

            # Test Base at I0
            self.assertEqual(calculate_price(item, I0), base, f"Base mismatch for {item}")

            # Test Scarcity at I0 - T
            calc_below = calculate_price(item, I0 - T)
            self.assertEqual(calc_below, p_below_T, f"P(I0-T) mismatch for {item}: got {calc_below}, expected {p_below_T}")

            # Test Glut at I0 + T
            calc_above = calculate_price(item, I0 + T)
            self.assertEqual(calc_above, p_above_T, f"P(I0+T) mismatch for {item}: got {calc_above}, expected {p_above_T}")

            # Test Glut at I0 + 2T
            calc_above_2T = calculate_price(item, I0 + 2 * T)
            self.assertEqual(calc_above_2T, p_above_2T, f"P(I0+2T) mismatch for {item}: got {calc_above_2T}, expected {p_above_2T}")

    def test_milk_price_impact(self):
        """Milk drops ~2.10 dollars per unit sold above I0."""
        p0 = calculate_price("MILK", 10000)
        self.assertEqual(p0, 160)
        p10 = calculate_price("MILK", 10010)
        # 160 - 10 * (256/122) = 160 - 20.98 = ~139
        self.assertEqual(p10, 139)

    def test_optimal_sale_batch(self):
        """Ensure selling large milk batch is truncated to avoid crashing price to floor."""
        # Starting at I0 with 50 units in shed
        batch = optimal_sale_batch("MILK", shed_qty=50, start_inv=10000, min_marginal_ratio=0.60)
        # 160 * 0.60 = 96.
        # Price drops below 96 around (160-96)/2.098 = 30.5 units
        self.assertTrue(25 <= batch <= 35, f"Unexpected batch size: {batch}")


if __name__ == "__main__":
    unittest.main()
