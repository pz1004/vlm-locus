#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
set -e
PY=${PY:-.venv/bin/python}
echo "=== 6-branch grid: calibration split ==="
$PY run/branches.py --ids runs/calib_ids.json --gamma 1.0 --out runs/branches6_calib.jsonl 2>&1 | tail -1
echo "=== 6-branch grid: test split ==="
$PY run/branches.py --gamma 1.0 --out runs/branches6_test.jsonl 2>&1 | tail -1
echo "=== STAGE3 DONE ==="
