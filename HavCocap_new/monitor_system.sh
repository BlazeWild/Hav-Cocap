#!/bin/bash

# Configuration
LOG_FILE="/home/ashok/Documents/blaze/Hav-Cocap/HavCocap_new/system_monitor.log"
DELAY=5 # Check every 5 seconds

echo "Starting System Monitor at $(date)" > "$LOG_FILE"
echo "Logging to $LOG_FILE"

while true; do
    echo "=================================================================" >> "$LOG_FILE"
    echo "TIMESTAMP: $(date)" >> "$LOG_FILE"
    
    echo -e "\n[SYSTEM MEMORY]" >> "$LOG_FILE"
    free -h >> "$LOG_FILE"
    
    echo -e "\n[SWAP USAGE]" >> "$LOG_FILE"
    swapon --show >> "$LOG_FILE"
    
    echo -e "\n[TOP 10 PROCESSES BY MEMORY]" >> "$LOG_FILE"
    ps -eo pid,user,%mem,%cpu,cmd --sort=-%mem | head -n 11 >> "$LOG_FILE"
    
    echo -e "\n[GPU UTILIZATION & MEMORY]" >> "$LOG_FILE"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv >> "$LOG_FILE"
    
    echo -e "\n[LATEST OOM (OUT OF MEMORY) KILLS]" >> "$LOG_FILE"
    # Suppress sudo requirement warnings for dmesg just in case, though usually works
    dmesg -T 2>/dev/null | grep -i -E 'killed process|oom' | tail -n 5 >> "$LOG_FILE"
    
    # Sync filesystem to ensure logs are written to disk immediately before a crash
    sync

    sleep $DELAY
done