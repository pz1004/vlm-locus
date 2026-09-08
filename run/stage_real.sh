#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
# Run the real-image families for one model. Difficulty was calibrated on Qwen2.5-VL-3B, so for
# any other model the levels are inherited rather than matched; model accuracy is reported
# alongside every probe number so a floor or ceiling effect is visible rather than hidden.
set -e
# every stage pipes through tail/grep for readability; without pipefail a failing python
# exits 0 and the whole run reports DONE having produced nothing
set -o pipefail
PY=${PY:-.venv/bin/python}
M=$1; T=$2; shift 2
for D in real_3b real_chart; do
  S=runs/states_${T}_${D}.npz
  echo "=== [$T/$D] behaviour ==="
  $PY run/score.py --model "$M" --data data/$D --out runs/${T}_${D}_gen.jsonl \
      --base-only --gen-only "$@" 2>&1 | tail -1
  echo "=== [$T/$D] capture ==="
  $PY run/capture.py --model "$M" --data data/$D --out $S "$@" 2>&1 | tail -1
  echo "=== [$T/$D] G1 ==="
  $PY run/probe.py $S 2>&1 | grep -viE 'warn|converg|explained_var'
  echo "=== [$T/$D] freeze probes ==="
  $PY run/fit_probes.py $S runs/probes_${T}_${D}.npy 2>&1 | grep -viE 'warn|converg'
  echo "=== [$T/$D] layer sweep ==="
  $PY run/layers.py $S runs/${T}_${D}_gen.jsonl runs/layers_${T}_${D}.json 2>&1 \
      | grep -viE 'warn|converg|explained_var'
  echo "=== [$T/$D] counterfactual ==="
  $PY run/cfprobe.py --model "$M" --data data/$D --probes runs/probes_${T}_${D}.npy \
      --prebuilt --res runs/cfprobe_${T}_${D}.json 2>&1 | grep -v "it/s" | tail -10
done
echo "=== ${T} REAL DONE ==="
