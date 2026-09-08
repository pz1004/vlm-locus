#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
set -e
PY=${PY:-.venv/bin/python}
$PY run/bon.py --k 8 --temp 1.0 --out runs/bon_t1_3b.jsonl 2>&1 | grep -v "it/s" | tail -12
$PY run/bon.py --k 5 --temp 0.7 --cot --out runs/boncot_3b.jsonl 2>&1 | grep -v "it/s" | tail -12
echo "=== STAGE4 DONE ==="
