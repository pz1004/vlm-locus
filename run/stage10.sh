#!/bin/bash
# Second architecture, identical data. The v1 SmolVLM run used its own difficulty-matched
# dataset, so its probe numbers are not comparable with Qwen's; this runs SmolVLM over the
# exact items Qwen was measured on. The correction library is skipped -- section 4 already
# showed no branch beats base -- so this measures only the claim that replicates or does not:
# is the attribute decodable from the states above what the model emits.
set -e
PY=${PY:-.venv/bin/python}
M=HuggingFaceTB/SmolVLM-Instruct
echo "=== SmolVLM behaviour on data/cal_3b ==="
$PY run/score.py --model $M --data data/cal_3b --out runs/smol_v2_gen.jsonl --base-only --gen-only 2>&1 | tail -2
echo "=== SmolVLM hidden states (eager) ==="
$PY run/capture.py --model $M --data data/cal_3b --out runs/states_smol_v2.npz 2>&1 | tail -2
echo "=== SmolVLM G1 presence test ==="
$PY run/probe.py runs/states_smol_v2.npz 2>&1 | grep -viE 'warn|converg|explained_var'
echo "=== SmolVLM layer sweep vs its own accuracy ==="
$PY run/layers.py runs/states_smol_v2.npz runs/smol_v2_gen.jsonl runs/layers_smol_v2.json \
    2>&1 | grep -viE 'warn|converg|explained_var'
echo "=== STAGE10 DONE ==="
