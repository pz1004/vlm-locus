#!/bin/bash
# Re-run the pipeline on the de-duplicated dataset. chart/counting/spatial regenerate
# bit-identically, so this is a tracking fix; the clean families are re-run only to prove
# they reproduce.
set -e
PY=${PY:-.venv/bin/python}
echo "=== capture states on the de-duplicated dataset ==="
$PY run/capture.py --data data/cal_3b_v2 --out runs/states_v2.npz 2>&1 | tail -2
echo "=== G1 presence test ==="
$PY run/probe.py runs/states_v2.npz 2>&1 | grep -viE 'warn|converg|explained_var'
echo "=== freeze probes ==="
$PY run/fit_probes.py runs/states_v2.npz runs/probes_v2.npy 2>&1 | grep -viE 'warn|converg'
$PY run/calib_ids.py runs/probes_v2.npy runs/states_v2_meta.json runs/calib_ids_v2.json
echo "=== branch grid, both splits ==="
$PY run/branches.py --data data/cal_3b_v2 --probes runs/probes_v2.npy \
    --ids runs/calib_ids_v2.json --gamma 1.0 --out runs/branches_v2_calib.jsonl 2>&1 | tail -1
$PY run/branches.py --data data/cal_3b_v2 --probes runs/probes_v2.npy \
    --gamma 1.0 --out runs/branches_v2_test.jsonl 2>&1 | tail -1
echo "=== STAGE6 DONE ==="
