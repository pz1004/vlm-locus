#!/bin/bash
# Wait on an explicit PID, then run the given script. Pattern matching on a command line
# matches the waiter itself, which is how an earlier waiter deadlocked; a PID cannot.
WAIT_PID=$1; shift
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 20; done
exec "$@"
