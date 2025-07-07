#!/bin/bash

# Function: Run 9 training commands in parallel, terminate all commands if any command fails

# Set error handling
set -e

# Define all commands
declare -a commands=(
    # "python src/baseline_h36m_30to30_pips/train.py"
    # "python src/baseline_h36m_30to30/train_no_traj.py"
    # "python src/baseline_h36m_15to15/train.py"
    # "python src/baseline_h36m_15to15/train_no_traj.py"
    # "python src/baseline_3dpw/train_norc.py"
    # "python src/baseline_3dpw/train_rc.py"
    # "python src/baseline_h36m_15to45/train.py"
    "python src/baseline_3dpw_big/train_rc.py"
    "python src/baseline_3dpw_big/train_norc.py"
)

# Define command descriptions
declare -a descriptions=(
    # "Mocap30to30"
    # "Mupots30to30"
    # "Mocap15to15"
    # "Mupots15to15"
    # "3dpw_norc"
    # "3dpw_rc"
    # "Mocap15to45"
    "3dpw_rc(pretrain)"
    "3dpw_norc(pretrain)"
)

# Store PIDs of all background processes
declare -a pids=()

# Cleanup function: Terminate all background processes
cleanup() {
    echo "Terminating all processes..."
    for pid in "${pids[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    # Wait for processes to fully terminate
    sleep 2
    
    # Force terminate running processes
    for pid in "${pids[@]}"; do
        if ps -p "$pid" > /dev/null 2>&1; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    echo "All processes terminated"
    exit 1
}

# Set signal handling
trap cleanup SIGINT SIGTERM

# Print prompt information
echo "====================================="
echo "Start running 9 training experiments in parallel"
echo "Use a single GPU to execute in parallel"
echo "====================================="
echo

# 启动所有命令
for i in "${!commands[@]}"; do
    echo "Start experiment $((i+1))/9: ${descriptions[$i]}"
    echo "Command: ${commands[$i]}"
    
    # Run command in background, no redirection
    ${commands[$i]} &
    
    # Record process ID
    pids+=($!)
    echo "Process ID: ${pids[$i]}"
    echo "---"
done

echo "All experiments started, monitoring process status..."
echo

# Monitor all processes
while true; do
    failed_count=0
    completed_count=0
    
    for i in "${!pids[@]}"; do
        pid=${pids[$i]}
        
        # Check if process is still running
        if ps -p "$pid" > /dev/null 2>&1; then
            # Process is still running
            continue
        else
            # Process has ended, check exit status
            if wait "$pid" 2>/dev/null; then
                # Process exited normally
                echo "✓ Experiment $((i+1)) (${descriptions[$i]}) completed successfully"
                completed_count=$((completed_count + 1))
            else
                # Process exited abnormally
                echo "✗ Experiment $((i+1)) (${descriptions[$i]}) failed"
                echo "Error information displayed above"
                failed_count=$((failed_count + 1))
                break
            fi
        fi
    done
    
    # If any process fails, terminate all other processes
    if [ $failed_count -gt 0 ]; then
        echo
        echo "Detected experiment failure, terminating all other processes..."
        cleanup
    fi
    
    # Check if all processes have completed
    if [ $completed_count -eq ${#commands[@]} ]; then
        echo
        echo "====================================="
        echo "All experiments completed successfully!"
        echo "====================================="
        break
    fi
    
    # Wait for a while before checking again
    sleep 5
done

echo "Script execution completed"
