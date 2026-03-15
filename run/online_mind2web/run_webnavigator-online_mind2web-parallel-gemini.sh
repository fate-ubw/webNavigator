#!/bin/bash

# ============================================
# Sharding parallel evaluation script - Gemini
# Usage: ./run_webnavigator-online_mind2web-parallel-gemini.sh [NUM_SHARDS]
# Example: ./run_webnavigator-online_mind2web-parallel-gemini.sh 8    # Start 8 parallel workers
# ============================================

NUM_SHARDS=${1:-1}  # Default: 1 parallel worker

mkdir -p logs
. run/env_init-95

echo "Starting $NUM_SHARDS parallel workers for Gemini..."

# Start N workers, each handling 1/N of tasks
for i in $(seq 0 $((NUM_SHARDS-1))); do
    python -u eval.py \
        --config configs/online_mind2web/webNavigator-env=online_mind2web-gemini.yml \
        --shard-index $i \
        --num-shards $NUM_SHARDS \
        2>&1 | tee logs/shard-gemini-$i.log &
    echo "  Started worker $i/$NUM_SHARDS (PID: $!)"
done

echo ""
echo "All $NUM_SHARDS workers launched. Check logs/shard-gemini-*.log for output."
echo "Use 'tail -f logs/shard-gemini-0.log' to monitor progress."

wait
echo "All parallel evaluations completed."
