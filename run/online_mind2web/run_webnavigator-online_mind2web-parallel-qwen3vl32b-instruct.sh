#!/bin/bash

# ============================================
# Parallel evaluation script - Qwen3-VL-32B-Instruct
# Usage: ./run_webnavigator-online_mind2web-parallel-qwen3vl32b-instruct.sh [NUM_WORKERS]
# Example: ./run_webnavigator-online_mind2web-parallel-qwen3vl32b-instruct.sh 16    # Start 16 parallel workers
# ============================================

NUM_WORKERS=${1:-8}  # Default: 8 parallel workers

mkdir -p logs
source run/online_mind2web/qwen_env_4090

echo "Starting $NUM_WORKERS parallel workers for Qwen3-VL-32B-Instruct..."

python eval_parallel.py \
    --config configs/online_mind2web/webNavigator-env=online_mind2web-qwen3vl32b-instruct.yml \
    --workers $NUM_WORKERS \
    2>&1 | tee logs/qwen3vl32b-instruct.log

echo "Evaluation completed. Check logs/qwen3vl32b-instruct.log for output."
