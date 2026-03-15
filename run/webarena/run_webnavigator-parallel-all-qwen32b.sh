NUM_SHARDS=${1:-1}  # Default: 1 parallel worker

mkdir -p logs
. run/env_init-webarena1
python browser_env/auto_login.py --env_name webarena1

CONFIG_FILE="configs/webarena/webNavigator-all-qwen32b.yml"
TRAJ_DIR="/home/nfs06/wtt/Trajectories/webNavigator-3_0-20260118-all-qwen32b"
SUMMARY_CSV="${TRAJ_DIR}/summary.csv"

# Keep track of all background worker PIDs
WORKER_PIDS=()

# Signal handler: clean up all background worker processes
cleanup_and_exit() {
    echo ""
    echo "=========================================="
    echo "Received termination signal. Cleaning up..."
    echo "=========================================="
    
    if [ ${#WORKER_PIDS[@]} -gt 0 ]; then
        echo "Stopping ${#WORKER_PIDS[@]} worker processes launched by this script..."
        
        # Try graceful termination first
        for pid in "${WORKER_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Stopping PID $pid..."
                kill "$pid" 2>/dev/null
            fi
        done
        
        # Wait for processes to finish
        sleep 2
        
        # Force-kill any remaining workers
        for pid in "${WORKER_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing PID $pid..."
                kill -9 "$pid" 2>/dev/null
            fi
        done
    else
        echo "No worker processes to clean up."
    fi
    
    echo "Cleanup completed. Exiting..."
    exit 130
}

# Catch Ctrl+C (SIGINT) and SIGTERM
trap cleanup_and_exit SIGINT SIGTERM

# Helper: clean tasks containing "You cannot post more" with success=0
cleanup_failed_tasks() {
    # Resolve the script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PYTHON_SCRIPT="${SCRIPT_DIR}/cleanup_failed_tasks.py"
    
    # Call the standalone Python script
    python3 "$PYTHON_SCRIPT" "$TRAJ_DIR"
    return $?
}

# Main evaluation function
run_evaluation() {
    echo "Starting $NUM_SHARDS parallel workers..."
    
    # Reset PID list before starting
    WORKER_PIDS=()
    
    # Start N workers, each handling 1/N of tasks
    for i in $(seq 0 $((NUM_SHARDS-1))); do
        python -u eval.py \
            --config "$CONFIG_FILE" \
            --shard-index $i \
            --num-shards $NUM_SHARDS \
            >> logs/shard-$i.log 2>&1 &
        worker_pid=$!
        WORKER_PIDS+=($worker_pid)
        echo "  Started worker $i/$NUM_SHARDS (PID: $worker_pid)"
    done
    
    echo ""
    echo "All $NUM_SHARDS workers launched. Check logs/shard-*.log for output."
    echo "Use 'tail -f logs/shard-0.log' to monitor progress."
    echo "Press Ctrl+C to stop all workers and exit."
    
    wait
    echo "All parallel evaluations completed."
    
    # Clear PID list after workers finish
    WORKER_PIDS=()
}

# Loop: run evaluation, clean failed tasks, rerun if cleaned tasks remain
ITERATION=1
while true; do
    echo ""
    echo "=========================================="
    echo "Iteration $ITERATION: Running evaluation..."
    echo "=========================================="
    
    # Run evaluation
    run_evaluation
    
    # Clean up failed tasks
    echo ""
    echo "=========================================="
    echo "Cleaning up failed tasks..."
    echo "=========================================="
    
    cleanup_failed_tasks
    cleanup_result=$?
    
    if [[ $cleanup_result -eq 0 ]]; then
        # Task cleanup completed; rerun needed
        echo ""
        echo "Some tasks were cleaned. Re-running evaluation with 1 worker..."
        ITERATION=$((ITERATION + 1))
        sleep 2  # Brief wait to ensure filesystem sync
    else
        # No cleanup performed
        if [[ $ITERATION -lt 20 ]]; then
            # Minimum iteration count not reached, continue running
            echo ""
            echo "=========================================="
            echo "No tasks cleaned, but iteration $ITERATION < 20."
            echo "Continuing to run evaluation..."
            echo "=========================================="
            NUM_SHARDS=1
            ITERATION=$((ITERATION + 1))
            sleep 2
        else
            # Minimum iterations reached with no cleanup; done
            echo ""
            echo "=========================================="
            echo "All tasks completed successfully!"
            echo "No more tasks with 'You cannot post more' and success=0."
            echo "Reached minimum 20 iterations."
            echo "=========================================="
            break
        fi
    fi
done

echo ""
echo "Final evaluation completed after $ITERATION iteration(s)."

