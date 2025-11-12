#!/bin/bash
echo "Monitoring weights directory..."
echo "Started at: $(date)"
echo ""

prev_size=0
prev_count=0
stable_count=0

while true; do
    if [ -d "/home/ubuntu/traced_model/MiniMax-M2/weights" ]; then
        size=$(du -sb /home/ubuntu/traced_model/MiniMax-M2/weights/ 2>/dev/null | cut -f1)
        count=$(find /home/ubuntu/traced_model/MiniMax-M2/weights/ -type f 2>/dev/null | wc -l)
        
        if [ "$size" != "$prev_size" ] || [ "$count" != "$prev_count" ]; then
            echo "[$(date +%H:%M:%S)] Weights: $count files, $(numfmt --to=iec $size)"
            prev_size=$size
            prev_count=$count
            stable_count=0
        else
            stable_count=$((stable_count + 1))
            if [ $stable_count -ge 10 ] && [ $count -gt 0 ]; then
                echo ""
                echo "=================================="
                echo "WEIGHTS SAVING COMPLETED!"
                echo "Time: $(date)"
                echo "Total files: $count"
                echo "Total size: $(numfmt --to=iec $size)"
                echo "=================================="
                break
            fi
        fi
    fi
    sleep 10
done
