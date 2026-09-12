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
echo "=== 4. G1 presence test ==="
$APY run/probe.py runs/states_3b.npz 2>&1 | grep -viE 'warn|converg'
echo "=== 5. freeze probes and steering directions ==="
$APY run/fit_probes.py 2>&1 | grep -viE 'warn|converg'
echo "=== 6. locus split ==="
$PY run/locus.py 2>&1 | grep -viE 'warn|converg'
echo "=== 7. G3 branches and oracle ==="
$PY run/branches.py --out runs/branches_3b.jsonl 2>&1 | grep -viE 'warn|Loading'
python3 run/oracle.py runs/branches_3b.jsonl
echo "=== STAGE2 DONE ==="
