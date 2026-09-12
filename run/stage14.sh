#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
# P2b: the real CHART family -- the load-bearing test.
#
# Chart is where the synthetic effect is concentrated (+64 pp, 89% follow, 98.6% decodable from
# the final layer). COCO gave spatial and counting cheaply and both came back negative; this is
# the family that decides whether that negative is about real images or about those two tasks.
#
# Source: ChartQA (Statista/Pew/OWID) with the underlying data tables. Admission required the
# detected bar heights to be an affine function of the table values, the table to be genuinely
# single-series, and the printed data labels to be strippable -- otherwise the task is OCR of a
# printed number rather than reading a bar against an axis.
set -e
PY=${PY:-.venv/bin/python}
# Analysis steps run under $APY, NOT $PY, and the separation is the point. The venv exists for
# torch; the committed analysis artefacts reproduce under the interpreter README.md's
# reproduction path names, and the two are not interchangeable -- numpy 2.4.6 and 2.5.2 disagree
# on 3 predictions of 1125 in this grid, one tie landing the other way in each of two cells.
# Producing a canonical artefact under whichever interpreter happened to have torch installed is
# how a committed file stops reproducing. run/verify_protocol.py asserts this separation.
APY=${APY:-python3}
D=data/real_chart

echo "=== 1. hidden states (eager) ==="
$PY run/capture.py --data $D --out runs/states_realchart.npz 2>&1 | tail -1
echo "=== 2. G1 presence test ==="
$APY run/probe.py runs/states_realchart.npz 2>&1 | grep -viE 'warn|converg|explained_var'
echo "=== 3. freeze probes ==="
$APY run/fit_probes.py runs/states_realchart.npz runs/probes_realchart.npy 2>&1 | grep -viE 'warn|converg'
echo "=== 4. layer sweep vs the model's own accuracy ==="
$APY run/layers.py runs/states_realchart.npz runs/realchart_gen.jsonl runs/layers_realchart.json 2>&1 \
    | grep -viE 'warn|converg|explained_var'
echo "=== 5. counterfactual control (prebuilt, exact) ==="
$PY run/cfprobe.py --data $D --probes runs/probes_realchart.npy --prebuilt \
    --res runs/cfprobe_realchart.json 2>&1 | grep -v "it/s" | tail -12
echo "=== STAGE14 DONE ==="
