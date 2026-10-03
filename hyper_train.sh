#!/bin/bash
set -e

GENS=${1:-20}
POP=${2:-4}

echo "=========================================================="
echo "Starting Double-Agent Self-Play Hyper-Training"
echo "Generations: $GENS | Population: $POP per archetype"
echo "Populations: Alpha (Livestock/Fertilizer) vs Beta (Cash Crops/Counter-Trader)"
echo "Benchmarks: Harry Taylor ($99k), GusNicho ($144k), Starter"
echo "Multi-core: 8 workers parallel execution"
echo "Auto-Resumes from checkpoint and auto-updates submission.py"
echo "=========================================================="

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$DIR:$PYTHONPATH"

caffeinate -i python3 evoexperiments/train_double_agent.py \
  --generations "$GENS" \
  --pop-size "$POP"

echo "=========================================================="
echo "Training session complete. Champion packaged in submission.tar.gz."
echo "=========================================================="
