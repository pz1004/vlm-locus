#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
# Regenerate every table in analysis/results-readout-3b.md from the frozen run artefacts.
# Analysis only -- no GPU, no model. Everything it reads was produced by stages 1-9.
set -e
PY=${PY:-.venv/bin/python}
Q='warn|explained_var|ConvergenceWarning'
echo "##### 1. dataset integrity (duplicate-render check)"
$PY gen/verify.py data/cal_3b 2>&1 | sed -n '/\[9\]/,/^$/p'
echo
echo "##### 2. where the answer lives: probe at every layer vs the model"
$PY run/layers.py 2>&1 | grep -viE "$Q"
echo
echo "##### 3. control: within-arm, where arm identity carries no information"
$PY run/generalize.py 2>&1 | grep -viE "$Q"
echo
echo "##### 4. control: counterfactual (causal)"
$PY -c "import json,sys; sys.path.insert(0,'run'); import cfprobe; cfprobe.report(json.load(open('runs/cfprobe_3b.json')))"
echo
echo "##### 5. control: presence, blindfold and shuffled-label"
$PY -c "
import json; d=json.load(open('runs/probe_g1.json'))
print(f\"{'family':10s}{'L':>3s}{'vis':>8s}{'blind':>8s}{'shuf':>8s}{'select':>8s}{'chance':>8s}\")
for f,v in d.items():
    print(f\"{f:10s}{v['layer']:3d}{100*v['vis']:7.1f}%{100*v['blind']:7.1f}%{100*v['shuffled']:7.1f}%\"
          f\"{100*v['selectivity']:7.1f}%{100*v['chance']:7.1f}%\")"
echo
echo "##### 6. master comparison: every method, one test split, paired"
$PY run/compare.py 2>&1 | grep -viE "$Q"
echo
echo "##### 7. does the routing method survive without labels"
$PY run/unsup_router.py 2>&1 | grep -viE "$Q"
echo
echo "##### 8. label budget: probe vs fine-tuning at matched supervision"
$PY run/datasize.py 2>&1 | grep -viE "$Q"
$PY run/datasize2.py 2>&1 | grep -viE "$Q"
echo
echo "##### 9. readout ceiling on the frozen states, and the gate"
$PY run/readout.py 2>&1 | grep -viE "$Q|Stochastic"
$PY run/gate.py 2>&1 | grep -viE "$Q"
echo
echo "##### 10. second model: SmolVLM on identical data"
$PY run/layers.py runs/states_smol_v2.npz runs/smol_v2_gen.jsonl runs/layers_smol_v2.json 2>&1 | grep -viE "$Q"
echo
echo "##### 11. second model, difficulty-matched"
$PY run/layers.py runs/states_smolm.npz runs/smolm_gen.jsonl runs/layers_smolm.json 2>&1 | grep -viE "$Q"
$PY -c "import json,sys; sys.path.insert(0,'run'); import cfprobe; cfprobe.report(json.load(open('runs/cfprobe_smolm.json')))"
echo
echo "##### 12. measured branch costs"
$PY run/tiered.py sec 2>&1 | sed -n '1,9p'
echo "##### REPORT DONE"
