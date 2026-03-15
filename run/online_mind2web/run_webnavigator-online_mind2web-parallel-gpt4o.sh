#!/bin/bash

# ============================================
# Sharding parallel evaluation script - GPT-4o
# Usage: ./run_webnavigator-online_mind2web-parallel-gpt4o.sh [NUM_SHARDS]
# Example: ./run_webnavigator-online_mind2web-parallel-gpt4o.sh 8    # Start 8 parallel workers
# ============================================

NUM_SHARDS=${1:-1}  # Default: 1 parallel worker

mkdir -p logs
. run/env_init-95

echo "Starting $NUM_SHARDS parallel workers for GPT-4o..."

# Start N workers, each handling 1/N of tasks
for i in $(seq 0 $((NUM_SHARDS-1))); do
    python -u eval.py \
        --config configs/online_mind2web/webNavigator-env=online_mind2web-gpt4o.yml \
        --shard-index $i \
        --num-shards $NUM_SHARDS \
        2>&1 | tee logs/shard-gpt4o-$i.log &
    echo "  Started worker $i/$NUM_SHARDS (PID: $!)"
done

echo ""
echo "All $NUM_SHARDS workers launched. Check logs/shard-gpt4o-*.log for output."
echo "Use 'tail -f logs/shard-gpt4o-0.log' to monitor progress."

wait
echo "All parallel evaluations completed."
