#!/bin/bash

# ============================================
# Sharding parallel evaluation script
# Usage: ./run_sharding.sh [NUM_SHARDS]
# Example: ./run_sharding.sh 8    # Start 8 parallel workers
# ============================================

NUM_SHARDS=${1:-1}  # Default: 1 parallel worker

mkdir -p logs
. run/env_init-webarena2
python browser_env/auto_login.py --env_name webarena2

echo "Starting $NUM_SHARDS parallel workers..."

# Start N workers, each handling 1/N of tasks
for i in $(seq 0 $((NUM_SHARDS-1))); do
    python -u eval.py \
        --config configs/webarena/webNavigator-all-gemini.yml \
        --shard-index $i \
        --num-shards $NUM_SHARDS \
        2>&1 | tee logs/shard-$i.log &
    echo "  Started worker $i/$NUM_SHARDS (PID: $!)"
done

echo ""
echo "All $NUM_SHARDS workers launched. Check logs/shard-*.log for output."
echo "Use 'tail -f logs/shard-0.log' to monitor progress."

wait
echo "All parallel evaluations completed."

