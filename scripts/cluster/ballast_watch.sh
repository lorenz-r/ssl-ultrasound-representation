#!/bin/bash
# Write a completion marker when a training chain exits, for any
# reason (completed, collapse-halt, or crash past MAX_ATTEMPTS).
#
# Why this exists: `ballast.py` releases its VRAM when it sees a marker in
# a watched log, but `train_ijepa.sh`'s own echoes go to the launcher's stdout,
# not to `wi17_ijepa_k4.log` (which receives only the Python process's output).
# So the ballast had nothing to watch. This closes that gap without touching
# the running job.
#
# Matched by command line rather than by PID: this outlives a multi-hour run and
# a recycled PID would otherwise end the ballast early.
set -u
SENTINEL=/mnt/home/benbarkow/logs/train_ijepa_sentinel.log
MARKER="=== WI-17 chain exited"

while pgrep -f "train_ijepa.sh" > /dev/null; do sleep 30; done
echo "$MARKER $(date '+%F %T')" >> "$SENTINEL"
