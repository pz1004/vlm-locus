#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
set -e
PY=${PY:-.venv/bin/python}
echo "=== 4. G1 presence test ==="
$PY run/probe.py runs/states_3b.npz 2>&1 | grep -viE 'warn|converg'
echo "=== 5. freeze probes and steering directions ==="
$PY run/fit_probes.py 2>&1 | grep -viE 'warn|converg'
echo "=== 6. locus split ==="
$PY run/locus.py 2>&1 | grep -viE 'warn|converg'
echo "=== 7. G3 branches and oracle ==="
$PY run/branches.py --out runs/branches_3b.jsonl 2>&1 | grep -viE 'warn|Loading'
python3 run/oracle.py runs/branches_3b.jsonl
echo "=== STAGE2 DONE ==="
