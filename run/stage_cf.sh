#!/bin/bash
# Capture counterfactual states at every layer, for every configuration, so that the follow rate
# can be computed at the final layer -- the same layer every other quantity in the locus verdict
# uses. run/cfprobe.py kept only the selected layer and saved nothing, which is why the published
# follow rates are the one mixed-estimator number in the paper.
set -e
set -o pipefail
PY=.venv/bin/python
Q3=Qwen/Qwen2.5-VL-3B-Instruct
Q7=Qwen/Qwen2.5-VL-7B-Instruct
IVL=OpenGVLab/InternVL3-2B-hf
SMOL=HuggingFaceTB/SmolVLM-Instruct

# tag | model | data | cf_root | extra flags
SPEC=(
  "3b|$Q3|data/cal_3b|data/cf_3b|"
  "real|$Q3|data/real_3b||"
  "q3b4_real_3b|$Q3|data/real_3b||--load-4bit"
  "q3b4_real_chart|$Q3|data/real_chart||--load-4bit"
  "q7b_real_3b|$Q7|data/real_3b||--load-4bit"
  "q7b_real_chart|$Q7|data/real_chart||--load-4bit"
  "ivl_real_3b|$IVL|data/real_3b||"
  "ivl_real_chart|$IVL|data/real_chart||"
  "smol_real_3b|$SMOL|data/real_3b||"
  "smol_real_chart|$SMOL|data/real_chart||"
  "smolm|$SMOL|data/cal_smolm||"
)
for row in "${SPEC[@]}"; do
  IFS='|' read -r T M D CFR EX <<< "$row"
  OUT=runs/cfstates_${T}.npz
  if [ -f "$OUT" ]; then echo "=== [$T] already captured, skipping ==="; continue; fi
  echo "=== [$T] counterfactual capture ==="
  ARGS=(--model "$M" --data "$D" --probes runs/probes_${T}.npy --out "$OUT")
  [ -n "$CFR" ] && ARGS+=(--cf-root "$CFR")
  [ -n "$EX" ] && ARGS+=("$EX")
  $PY run/cfcapture.py "${ARGS[@]}" 2>&1 | grep -vE "it/s|Loading weights|^Warning" | tail -3
done
echo "=== CF CAPTURE DONE ==="
