#!/bin/bash
echo "Monitoring compilation progress..."
echo "Started at: $(date)"
echo ""

while true; do
    # Check if state_dict loading has started
    if grep -q "CUSTOM get_state_dict\|Converting FP8\|Block-wise scale" compilation_output.log 2>/dev/null; then
        echo ""
        echo "=================================="
        echo "CHECKPOINT LOADING HAS STARTED!"
        echo "Time: $(date)"
        echo "=================================="
        echo ""
        grep -E "CUSTOM get_state_dict|Converting FP8|Renaming attention|Block-wise|Total converted|Total renamed" compilation_output.log | tail -20
        break
    fi
    
    # Check compilation progress
    if grep -q "Compiler status" compilation_output.log 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] Compilation in progress..."
        tail -5 compilation_output.log | grep "Compiler status" | tail -1
    fi
    
    sleep 30
done
