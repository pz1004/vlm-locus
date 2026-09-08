#!/bin/bash
# Fine-tuning gain per (model, family), to test the §7 claim -- that probe accuracy lower-bounds
# and rank-orders what fine-tuning achieves -- at n=32 pairs instead of n=4.
#
# Trained on the probe's own train_ids and evaluated on the probe's own test_ids, so probe and
# fine-tune see identical data. --eval-base scores the base model through the same code path, so
# the gain is a difference between two numbers produced by one scorer.
set -e; set -o pipefail
PY=.venv/bin/python
M=$1; T=$2; shift 2
# the module set must be part of the tag: without it a --modules lang run silently
# overwrites the per-item files of the --modules both run for the same model
MODTAG=""
for x in "$@"; do case "$x" in lang|vis) MODTAG="_$x";; esac; done
for D in real_3b real_chart; do
  P=runs/probes_${T}_${D}.npy; I=runs/testids_${T}_${D}.jsonl
  [ -f "$P" ] || { P=runs/probes_${D/real_3b/real}.npy; I=runs/testids_${D/real_3b/real}.jsonl; }
  [ "$D" = real_chart ] && [ ! -f "runs/probes_${T}_${D}.npy" ] && \
      { P=runs/probes_realchart.npy; I=runs/testids_realchart.jsonl; }
  echo "=== LoRA [$T/$D]  probes=$P"
  $PY run/lora.py --model "$M" --data data/$D --probes "$P" --test "$I" \
      --out runs/lora_${T}_${D}${MODTAG}.json --tag lora_${T}_${D}${MODTAG} --eval-base "$@" 2>&1 \
      | grep -E "^  \[|epoch|adapted|note:|wrote|train " || true
done
echo "=== LORA ${T} DONE ==="
