#!/bin/bash
# Promote the de-duplicated, eager-captured run to canonical, then refresh everything that
# depends on it. chart/counting/spatial images are bit-identical between v1 and v2, so only
# tracking needs new label-free features; the probes are refit for all four families because
# the capture moved from sdpa to eager.
set -e
PY=${PY:-.venv/bin/python}
mkdir -p runs/v1_contaminated
for f in states_3b.npz states_3b_meta.json probe_g1.json probes_3b.npy calib_ids.json \
         branches6_calib.jsonl branches6_test.jsonl; do
  [ -e "runs/$f" ] && mv "runs/$f" "runs/v1_contaminated/$f"
done
mv runs/states_v2.npz runs/states_3b.npz
mv runs/states_v2_meta.json runs/states_3b_meta.json
mv runs/probe_g1_v2.json runs/probe_g1.json
mv runs/probes_v2.npy runs/probes_3b.npy
mv runs/calib_ids_v2.json runs/calib_ids.json
mv runs/branches_v2_calib.jsonl runs/branches6_calib.jsonl
mv runs/branches_v2_test.jsonl runs/branches6_test.jsonl
rm -rf data/cal_3b_v1 && mv data/cal_3b data/cal_3b_v1 && mv data/cal_3b_v2 data/cal_3b
echo "=== counterfactual control (all families, new probes) ==="
$PY run/cfprobe.py 2>&1 | grep -v "it/s" | tail -14
echo "=== label-free features: tracking only (other families' data is bit-identical) ==="
cp runs/unsup_calib_3b.jsonl runs/v1_contaminated/ 2>/dev/null || true
cp runs/unsup_test_3b.jsonl  runs/v1_contaminated/ 2>/dev/null || true
grep -v '"family": "tracking"' runs/v1_contaminated/unsup_calib_3b.jsonl > runs/unsup_calib_3b.jsonl
grep -v '"family": "tracking"' runs/v1_contaminated/unsup_test_3b.jsonl  > runs/unsup_test_3b.jsonl
$PY run/unsup.py --families tracking --ids runs/calib_ids.json --out runs/_u_cal_tr.jsonl 2>&1 | tail -1
$PY run/unsup.py --families tracking --out runs/_u_te_tr.jsonl 2>&1 | tail -1
cat runs/_u_cal_tr.jsonl >> runs/unsup_calib_3b.jsonl && rm runs/_u_cal_tr.jsonl
cat runs/_u_te_tr.jsonl  >> runs/unsup_test_3b.jsonl  && rm runs/_u_te_tr.jsonl
echo "=== STAGE7 DONE ==="
