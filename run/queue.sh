#!/bin/bash
# wait on an explicit PID -- pattern matching on the command line matches this script itself
WAIT_PID=$1
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
echo "=== cfprobe: counterfactual control ==="
.venv/bin/python run/cfprobe.py 2>&1 | grep -v "it/s"
echo "=== stage5: label-free features ==="
bash run/stage5.sh
echo "=== QUEUE DONE ==="
