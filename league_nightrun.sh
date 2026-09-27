#!/bin/bash
set -e

echo "=========================================================="
echo "Starting Kaggriculture Multi-Agent Co-Evolution League"
echo "Archetypes: SwarmPlanter vs DairyBaron"
echo "Stress Opponents: Harry Taylor (113441225), Starter"
echo "Profit Ratchet Floor: Progressive ($70k -> $100k+)"
echo "Generations: 60 | Population: 4 per archetype"
echo "Log file: league_train.log"
echo "Checkpoints saved to: evoexperiments/checkpoints/best_league_champion.json"
echo "=========================================================="

export PYTHONPATH="/Users/apple/Desktop/kragliculture:$PYTHONPATH"

caffeinate -i python3 evoexperiments/train_league.py \
  --generations 600 \
  --population 4 \
  | tee league_train.log
