#!/bin/bash
set -e
PY=${PY:-.venv/bin/python}
$PY run/unsup.py --ids runs/calib_ids.json --out runs/unsup_calib_3b.jsonl 2>&1 | tail -2
$PY run/unsup.py --out runs/unsup_test_3b.jsonl 2>&1 | tail -2
echo "=== STAGE5 DONE ==="
