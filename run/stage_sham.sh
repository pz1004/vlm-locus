#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
#
# The label-preserving control, end to end. Builds the sham chart edits, verifies them, scores
# each model on them and captures the counterfactual states, then reports stability.
#
# The sham raises a bar the question does not ask about, by the same amount as that item's real
# counterfactual, so "the probe follows the real edit and not the sham" cannot be explained by
# the sham being a smaller change. Named here rather than left to an ad-hoc invocation: an
# artefact whose producer names it only through argv is findable by nothing, which is how
# runs/canonpred_*.json came to look producerless. verify_protocol.py check 19 asserts it.
set -e
set -o pipefail
cd "$(dirname "$0")/.."
PY=$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )
D=data/real_chart_sham

$PY gen/build_chart_sham.py --src data/real_chart_v2 --out $D
$PY gen/verify_sham.py --root $D

run() {  # tag model extra-flags [max-new]
  T=$1; M=$2; X=$3; N=${4:-12}
  echo "=== [$T] score sham ==="
  $PY run/score.py --model "$M" --data $D --out "runs/${T}_sham_gen.jsonl" --gen-only \
      --max-new "$N" $X 2>&1 | tail -1
  $PY run/fix_chart_scoring.py "runs/${T}_sham_gen.jsonl" 2>&1 | tail -1
  echo "=== [$T] capture sham counterfactual states ==="
  $PY run/cfcapture.py --model "$M" --data $D --probes "runs/probes_${T}.npy" \
      --out "runs/cfstates_${T}_sham.npz" $X 2>&1 | tail -1
}

run realchart_v2        Qwen/Qwen2.5-VL-3B-Instruct    ""
run q3b4_real_chart_v2  Qwen/Qwen2.5-VL-3B-Instruct    --load-4bit
run q7b_real_chart_v2   Qwen/Qwen2.5-VL-7B-Instruct    --load-4bit 48
run ivl_real_chart_v2   OpenGVLab/InternVL3-2B-hf      ""
run smol_real_chart_v2  HuggingFaceTB/SmolVLM-Instruct ""

$PY run/shamfollow.py --out runs/sham.json
