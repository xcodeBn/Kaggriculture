#!/bin/bash
set -e

echo "=========================================================="
echo "Starting Kaggriculture Overnight Recursive Training"
echo "Opponents: Harry Taylor (113441225), GusNicho (113076092), Starter, Random"
echo "Generations: 150 | Population: 8"
echo "Log file: train_overnight.log"
echo "Checkpoints saved to: evoexperiments/checkpoints/best_config.json"
echo "=========================================================="

export PYTHONPATH="/Users/apple/Desktop/kragliculture:$PYTHONPATH"

caffeinate -i python3 evoexperiments/train.py \
  --generations 150 \
  --population 8 \
  --opponents \
    replay:replays/kaggle_replays/113441225.json:1 \
    replay:replays/kaggle_replays/113076092.json:1 \
    starter \
    random \
  | tee train_overnight.log
