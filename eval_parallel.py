#!/usr/bin/env python3
"""
Online Mind2Web parallel evaluation script

Features:
1. Execute tasks in parallel with a process pool
2. Automatically retry failed tasks up to 3 times
3. Graceful Ctrl+C shutdown with child-process cleanup
4. Periodic cleanup of orphaned Xvfb processes
5. Resume from checkpoint by skipping already completed tasks

Usage:
    python eval_online_mind2web_parallel.py --config <config.yml> --workers 8
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import yaml
import json
import threading
import psutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()


# Global variables used by signal handling
_shutdown_requested = False
_main_pid = None


def get_dstdir(config_path):
    """Get destination directory from config (same logic as eval.py)."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logdir = config.get('logdir', '../Trajectories')
    logname = config.get('logname', '')
    
    # Get model name
    agent_config = config.get('agent', {})
    actor_config = agent_config.get('actor', {})
    model = actor_config.get('model', agent_config.get('model_name', ''))
    model_name = model.split('/')[-1] if model else ''
    
    if logname:
        if model_name and model_name not in logname:
            dstdir = f"{logdir}/{logname}-{model_name}"
        else:
            dstdir = f"{logdir}/{logname}"
    else:
        if model_name:
            dstdir = f"{logdir}/{datetime.now().strftime('%Y%m%d-%H%M%S')}-{model_name}"
        else:
            dstdir = f"{logdir}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    return dstdir


def get_task_ids(config_path):
    """Get task IDs from config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    benchmark_name = config.get('benchmark', {}).get('name', 'online_mind2web')
    task_name = config.get('benchmark', {}).get('task_name', 'task')
    task_ids = config.get('env', {}).get('task_ids', [])
    
    if task_ids == "all" or task_ids == ["all"]:
        task_dir = f"config_files/{benchmark_name}/{task_name}"
        task_ids = [f[:-5] for f in os.listdir(task_dir) if f.endswith('.json')]
        task_ids.sort(key=lambda x: int(x) if x.isdigit() else x)
    
    return [str(tid) for tid in task_ids]


def get_completed_tasks(dstdir):
    """Get list of completed task IDs."""
    completed = set()
    if os.path.exists(dstdir):
        for f in os.listdir(dstdir):
            if f.endswith('.json') and f != 'summary.csv':
                task_id = f[:-5]  # Strip the .json suffix
                if task_id.isdigit():
                    completed.add(task_id)
    return completed


def kill_zombie_xvfb():
    """Clean up orphan Xvfb processes (processes with ppid == 1).
    
    Equivalent to: ps -eo pid,ppid,cmd | grep Xvfb | grep -v grep | awk '$2==1 {print $1}' | xargs kill
    """
    killed = 0
    for proc in psutil.process_iter(['pid', 'ppid', 'name']):
        try:
            # Keep only Xvfb processes with ppid == 1 (orphaned by init)
            if 'Xvfb' in proc.info['name'] and proc.info['ppid'] == 1:
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return killed


def xvfb_cleaner_thread(stop_event, interval=60):
    """Background thread that periodically removes zombie Xvfb processes."""
    while not stop_event.is_set():
        killed = kill_zombie_xvfb()
        if killed > 0:
            print(f"[Xvfb Cleaner] Killed {killed} zombie Xvfb processes")
        stop_event.wait(interval)


def run_single_task(config_path, task_id, worker_id, timeout):
    """
    Run a single task in a worker subprocess.
    Returns: (task_id, success, error_message)
    
    A task is considered successful when {task_id}.json is saved to output directory.
    """
    # Ignore SIGINT in worker subprocess so main process handles shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # Switch to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create a temporary config containing only this task
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get output directory and result file path for completion checks
    dstdir = get_dstdir(config_path)
    result_file = os.path.join(dstdir, f"{task_id}.json")
    
    # Track whether result file existed before execution (to avoid false positives)
    result_existed_before = os.path.exists(result_file)
    result_mtime_before = os.path.getmtime(result_file) if result_existed_before else 0
    
    # Update config to run only one task
    config['env']['task_ids'] = [int(task_id) if task_id.isdigit() else task_id]
    
    temp_config_path = f"/tmp/task_{task_id}_worker_{worker_id}_{os.getpid()}.yml"
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f)
    
    try:
        # Run eval.py and stream output directly to terminal
        result = subprocess.run(
            [sys.executable, "-u", "eval.py", "--config", temp_config_path],
            cwd=script_dir,
            timeout=timeout
        )
        
        # Verify success by checking whether {task_id}.json is written.
        if os.path.exists(result_file):
            # Confirm the file is newly created/updated in this run.
            result_mtime_after = os.path.getmtime(result_file)
            if result_mtime_after > result_mtime_before:
                return (task_id, True, None)
            else:
                # File exists but was not updated; could be leftover from previous run.
                return (task_id, False, f"Result file not updated (exit code: {result.returncode})")
        else:
            return (task_id, False, f"Result file not created (exit code: {result.returncode})")
            
    except subprocess.TimeoutExpired:
        return (task_id, False, f"Task timeout ({timeout}s)")
    except Exception as e:
        return (task_id, False, str(e))
    finally:
        # Remove temporary config file
        if os.path.exists(temp_config_path):
            try:
                os.remove(temp_config_path)
            except:
                pass


def cleanup_all_children():
    """Terminate all child processes."""
    current_pid = os.getpid()
    try:
        parent = psutil.Process(current_pid)
        children = parent.children(recursive=True)
        
        # Send SIGTERM first
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Wait up to 5 seconds for children to exit
        gone, alive = psutil.wait_procs(children, timeout=5)
        
        # Send SIGKILL to remaining processes
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
                
    except psutil.NoSuchProcess:
        pass
    
    # Final Xvfb cleanup
    killed = kill_zombie_xvfb()
    if killed > 0:
        print(f"[Cleanup] Killed {killed} orphan Xvfb processes")


def main():
    global _shutdown_requested, _main_pid
    
    parser = argparse.ArgumentParser(
        description="Online Mind2Web parallel evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python eval_online_mind2web_parallel.py --config configs/config.yml --workers 8
    python eval_online_mind2web_parallel.py --config configs/config.yml --workers 4 --max-retries 5
        """
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=4, 
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--max-retries", 
        type=int, 
        default=3, 
        help="Maximum retries per task (default: 3)"
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=21600, 
        help="Per-task timeout in seconds (default: 21600, i.e. 6 hours)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Record main process PID
    _main_pid = os.getpid()
    
    config_path = os.path.abspath(args.config)
    num_workers = args.workers
    max_retries = args.max_retries
    task_timeout = args.timeout
    
    # Get output directory and task metadata
    dstdir = get_dstdir(config_path)
    all_task_ids = get_task_ids(config_path)
    completed_tasks = get_completed_tasks(dstdir)
    
    # Filter out already completed tasks
    pending_tasks = [tid for tid in all_task_ids if tid not in completed_tasks]
    
    print(f"[Init] Total tasks: {len(all_task_ids)}")
    print(f"[Init] Completed tasks: {len(completed_tasks)}")
    print(f"[Init] Pending tasks: {len(pending_tasks)}")
    print(f"[Init] Output directory: {dstdir}")
    print(f"[Init] Workers: {num_workers}")
    print(f"[Init] Max retries per task: {max_retries}")
    print(f"[Init] Task timeout: {task_timeout}s ({task_timeout//3600}h {(task_timeout%3600)//60}m)")
    
    if not pending_tasks:
        print("[Done] All tasks already completed!")
        return
    
    # Task queue and counters
    task_queue = list(pending_tasks)
    retry_counts = defaultdict(int)
    completed = set()
    failed = set()
    
    # Ensure output directory exists
    os.makedirs(dstdir, exist_ok=True)
    
    # Remove leftover temporary config files from previous runs
    if os.path.exists(dstdir):
        temp_files = [f for f in os.listdir(dstdir) if f.startswith('task_') and f.endswith('.yml')]
        if temp_files:
            print(f"[Cleanup] Removing {len(temp_files)} leftover temp config files...")
            for f in temp_files:
                try:
                    os.remove(os.path.join(dstdir, f))
                except:
                    pass
    
    # Start Xvfb cleanup thread
    cleaner_stop = threading.Event()
    cleaner_thread = threading.Thread(
        target=xvfb_cleaner_thread, 
        args=(cleaner_stop, 60),
        daemon=True
    )
    cleaner_thread.start()
    
    executor = None
    
    try:
        # Use ProcessPoolExecutor and ignore SIGINT in children
        executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
        )
        
        futures = {}
        worker_id = 0
        
        # Submit initial batch of tasks
        while len(futures) < num_workers and task_queue:
            task_id = task_queue.pop(0)
            future = executor.submit(run_single_task, config_path, task_id, worker_id, task_timeout)
            futures[future] = task_id
            worker_id += 1
            print(f"[Submit] Task {task_id} submitted (worker {worker_id})")
        
        # Main loop: wait for completed tasks and schedule replacements
        while futures and not _shutdown_requested:
            # Wait for any task to finish
            done_futures = []
            
            # Check whether any task is finished
            for future in list(futures.keys()):
                if future.done():
                    done_futures.append(future)
            
            # If none completed yet, wait briefly and check again.
            if not done_futures:
                time.sleep(0.5)
                continue
            
            # Handle finished tasks
            for future in done_futures:
                if _shutdown_requested:
                    break
                    
                task_id = futures.pop(future)
                try:
                    result_task_id, success, error = future.result()
                    
                    if success:
                        completed.add(result_task_id)
                        print(f"[Done] Task {result_task_id} completed successfully")
                    else:
                        print(f"[Error] Task {result_task_id}: {error}")
                        retry_counts[result_task_id] += 1
                        if retry_counts[result_task_id] < max_retries:
                            print(f"[Retry] Task {result_task_id} will retry ({retry_counts[result_task_id]}/{max_retries})")
                            task_queue.append(result_task_id)
                        else:
                            print(f"[Failed] Task {result_task_id} exceeded max retries")
                            failed.add(result_task_id)
                except Exception as e:
                    print(f"[Error] Task {task_id} raised exception: {e}")
                    retry_counts[task_id] += 1
                    if retry_counts[task_id] < max_retries:
                        task_queue.append(task_id)
                    else:
                        failed.add(task_id)
                
                # Submit a new task if any are pending
                if task_queue and not _shutdown_requested:
                    new_task_id = task_queue.pop(0)
                    new_future = executor.submit(run_single_task, config_path, new_task_id, worker_id, task_timeout)
                    futures[new_future] = new_task_id
                    worker_id += 1
                    print(f"[Submit] Task {new_task_id} submitted")
                
                # Print progress
                print(f"[Progress] Active: {len(futures)}, Completed: {len(completed)}, "
                      f"Failed: {len(failed)}, Pending: {len(task_queue)}")
    
    except KeyboardInterrupt:
        print("\n[Interrupt] Ctrl+C received, shutting down...")
        _shutdown_requested = True
    
    finally:
        # Stop the Xvfb cleanup thread
        cleaner_stop.set()
        
        # Shut down the process pool
        if executor:
            print("[Cleanup] Shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
        
        # Terminate all child processes
        print("[Cleanup] Terminating child processes...")
        cleanup_all_children()
        
        # Print final statistics
        print(f"\n{'='*50}")
        print("Final Statistics")
        print(f"{'='*50}")
        print(f"  Completed: {len(completed)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Pending: {len(task_queue)}")
        if failed:
            print(f"  Failed tasks: {sorted(failed, key=lambda x: int(x) if x.isdigit() else x)}")
        print(f"  Output directory: {dstdir}")


if __name__ == "__main__":
    main()
