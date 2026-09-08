#!/bin/bash
# The sampling baselines were measured on the pre-dedup tracking items, whose images no longer
# exist. chart/counting/spatial regenerate bit-identically so their rows stand; only tracking
# is re-sampled and spliced back in.
set -e
PY=${PY:-.venv/bin/python}
mkdir -p runs/v1_contaminated
for tag in "bon_3b:--temp 0.7" "bon_t1_3b:--temp 1.0" "boncot_3b:--cot --temp 0.7"; do
  f=${tag%%:*}; args=${tag#*:}
  echo "=== $f  ($args) tracking re-run ==="
  cp "runs/$f.jsonl" "runs/v1_contaminated/$f.jsonl"
  grep -v '"family": "tracking"' "runs/v1_contaminated/$f.jsonl" > "runs/$f.jsonl"
  $PY run/bon.py --families tracking --out "runs/_t_$f.jsonl" $args 2>&1 | tail -1
  cat "runs/_t_$f.jsonl" >> "runs/$f.jsonl" && rm "runs/_t_$f.jsonl"
  echo "  $(wc -l < runs/$f.jsonl) rows"
done
echo "=== STAGE8 DONE ==="
