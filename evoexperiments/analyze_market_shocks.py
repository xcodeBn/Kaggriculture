"""Analyze price and inventory movement around sales in Kaggle replays."""
import json
from pathlib import Path
from collections import defaultdict

replays_dir = Path("replays/kaggle_replays")
replays = sorted(replays_dir.glob("*.json"))

TARGET_PRODUCTS = {"MILK", "WOOL", "MELON", "STRAWBERRY"}

def analyze_replays():
    print(f"Analyzing {len(replays)} replay files for market price shocks...")
    
    total_large_sales = 0
    price_collapses = 0
    sales_by_product = defaultdict(list)
    
    for rpath in replays:
        if rpath.name == "kaggle_game.json":
            continue # duplicate of 113076092.json
            
        with open(rpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        steps = data.get("steps", [])
        if len(steps) < 10:
            continue
            
        # Track market price and inventory across frames
        for f_idx in range(1, len(steps)):
            # Observation at frame f_idx is after actions from frame f_idx executed
            # Frame f_idx has the actions submitted at step f_idx - 1
            frame = steps[f_idx]
            prev_obs = steps[f_idx - 1][0].get("observation", {})
            curr_obs = steps[f_idx][0].get("observation", {})
            
            prev_prices = prev_obs.get("market", {}).get("prices", {})
            curr_prices = curr_obs.get("market", {}).get("prices", {})
            
            prev_inv = prev_obs.get("market", {}).get("inventory", {})
            curr_inv = curr_obs.get("market", {}).get("inventory", {})
            
            # Check actions from both players in this frame
            for p_idx in [0, 1]:
                action = frame[p_idx].get("action") or {}
                market_orders = action.get("market") or []
                for order in market_orders:
                    if isinstance(order, (list, tuple)) and len(order) >= 3 and order[0] == "SELL":
                        prod = order[1]
                        qty = int(order[2])
                        if prod in TARGET_PRODUCTS and qty >= 5:
                            p_before = prev_prices.get(prod)
                            p_after = curr_prices.get(prod)
                            inv_before = prev_inv.get(prod)
                            inv_after = curr_inv.get(prod)
                            
                            total_large_sales += 1
                            if p_before and p_after and p_after <= 2:
                                price_collapses += 1
                                
                            sales_by_product[prod].append({
                                "replay": rpath.name,
                                "frame": f_idx,
                                "player": p_idx,
                                "qty_requested": qty,
                                "price_before": p_before,
                                "price_after": p_after,
                                "inv_before": inv_before,
                                "inv_after": inv_after,
                            })

    print(f"\nTotal large sales (qty >= 5 of MILK/WOOL/MELON/STRAWBERRY): {total_large_sales}")
    print(f"Sales resulting in floor collapse (price <= $2): {price_collapses}")
    
    print("\nBreakdown by Product:")
    for prod, records in sorted(sales_by_product.items()):
        print(f"\n--- {prod} ({len(records)} large sales) ---")
        # Show top 5 largest drops
        drops = [r for r in records if r["price_before"] is not None and r["price_after"] is not None]
        drops.sort(key=lambda x: (x["price_before"] - x["price_after"]), reverse=True)
        for d in drops[:5]:
            p_diff = d["price_before"] - d["price_after"]
            inv_diff = (d["inv_after"] - d["inv_before"]) if d["inv_before"] and d["inv_after"] else "?"
            print(f"  [{d['replay']} frame {d['frame']} P{d['player']}] SELL {prod} {d['qty_requested']}: "
                  f"Price {d['price_before']} -> {d['price_after']} (Δ={p_diff}), Inv {d['inv_before']} -> {d['inv_after']} (Δ={inv_diff})")

if __name__ == "__main__":
    analyze_replays()
