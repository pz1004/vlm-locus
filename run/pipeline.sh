#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
set -e
PY=${PY:-.venv/bin/python}
# Analysis steps run under $APY, NOT $PY, and the separation is the point. The venv exists for
# torch; the committed analysis artefacts reproduce under the interpreter README.md's
# reproduction path names, and the two are not interchangeable -- numpy 2.4.6 and 2.5.2 disagree
# on 3 predictions of 1125 in this grid, one tie landing the other way in each of two cells.
# Producing a canonical artefact under whichever interpreter happened to have torch installed is
# how a committed file stops reproducing. run/verify_protocol.py asserts this separation.
APY=${APY:-python3}
LV=$(cat runs/levels_3b.json | tr -d '\n ')
echo "=== 1. generate at locked levels $LV ==="
rm -rf data/cal_3b
python3 gen/generate.py --out data/cal_3b --n 300 --seed 20260905 --no-cf --levels "$LV"
python3 gen/verify.py data/cal_3b | tail -2
echo "=== 2. behaviour (generation) ==="
$PY run/score.py --data data/cal_3b --out runs/cal_3b_gen.jsonl --base-only --gen-only
echo "=== 3. hidden states, real + blindfold ==="
$PY run/capture.py --data data/cal_3b --out runs/states_3b.npz
echo "=== 4. G1: presence test ==="
$APY run/probe.py runs/states_3b.npz 2>&1 | grep -viE 'warn|converg'
echo "=== 5. freeze probes + steering directions ==="
$APY run/fit_probes.py 2>&1 | grep -viE 'warn|converg'
echo "=== 6. locus split ==="
$PY run/locus.py 2>&1 | grep -viE 'warn|converg'
echo "=== 7. G3: branches + oracle ==="
$PY run/branches.py --out runs/branches_3b.jsonl
python3 run/oracle.py runs/branches_3b.jsonl
echo "=== PIPELINE DONE ==="
