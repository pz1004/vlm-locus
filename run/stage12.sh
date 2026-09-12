#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
# Difficulty-matched SmolVLM replication.
#
# Section 10 ran SmolVLM on data calibrated for Qwen, where it sits near chance on spatial and
# tracking. Floor effects compress a probe-vs-model gap from both ends, so its smaller numbers
# there are a lower bound rather than a measurement. This repeats the whole protocol at
# SmolVLM's *own* difficulty levels, chosen the same way Qwen's were: the level per family at
# which the model is closest to 50% wrong.
#
# The generator is the fixed one, so the sweep and the calibration set are both de-duplicated;
# tracking at a low level has a small spec space and build() will say so rather than repeat
# images.
set -e
PY=${PY:-.venv/bin/python}
# Analysis steps run under $APY, NOT $PY, and the separation is the point. The venv exists for
# torch; the committed analysis artefacts reproduce under the interpreter README.md's
# reproduction path names, and the two are not interchangeable -- numpy 2.4.6 and 2.5.2 disagree
# on 3 predictions of 1125 in this grid, one tie landing the other way in each of two cells.
# Producing a canonical artefact under whichever interpreter happened to have torch installed is
# how a committed file stops reproducing. run/verify_protocol.py asserts this separation.
APY=${APY:-python3}
M=HuggingFaceTB/SmolVLM-Instruct

echo "=== 1. level-swept set (fixed generator) ==="
rm -rf data/sweep_smolm
python3 gen/generate.py --out data/sweep_smolm --n 200 --seed 20260907 --no-cf --levels sweep | tail -6

echo "=== 2. SmolVLM difficulty sweep ==="
$PY run/score.py --model $M --data data/sweep_smolm --out runs/sweep_smolm.jsonl \
    --base-only --gen-only 2>&1 | tail -1
python3 - <<'PY'
import json, numpy as np
rs = [json.loads(l) for l in open('runs/sweep_smolm.jsonl')]
best = {}
print(f"{'family':10s} " + ''.join(f'{f"L{l}":>9s}' for l in range(5)) + '   -> lock')
for f in sorted({r['family'] for r in rs}):
    row = [100 * (1 - np.mean([x['gen_correct'] for x in rs
                               if x['family'] == f and x['level'] == l])) for l in range(5)]
    cand = [l for l in range(5) if 40 <= row[l] <= 60]
    pick = cand[len(cand) // 2] if cand else int(np.argmin([abs(e - 50) for e in row]))
    best[f] = pick
    print(f'{f:10s} ' + ''.join(f'{e:8.0f}%' for e in row) + f'   L{pick} ({row[pick]:.0f}%)')
json.dump(best, open('runs/levels_smolm.json', 'w'), indent=1)
PY

LV=$(tr -d '\n ' < runs/levels_smolm.json)
echo "=== 3. calibration set at SmolVLM's levels $LV ==="
rm -rf data/cal_smolm
python3 gen/generate.py --out data/cal_smolm --n 300 --seed 20260907 --levels "$LV" | tail -6
$PY gen/verify.py data/cal_smolm 2>&1 | sed -n '/\[9\]/,/^$/p'

echo "=== 4. behaviour ==="
$PY run/score.py --model $M --data data/cal_smolm --out runs/smolm_gen.jsonl \
    --base-only --gen-only 2>&1 | tail -1
echo "=== 5. hidden states (eager) ==="
$PY run/capture.py --model $M --data data/cal_smolm --out runs/states_smolm.npz 2>&1 | tail -1
echo "=== 6. G1 presence test ==="
$APY run/probe.py runs/states_smolm.npz 2>&1 | grep -viE 'warn|converg|explained_var'
echo "=== 7. freeze probes ==="
$APY run/fit_probes.py runs/states_smolm.npz runs/probes_smolm.npy 2>&1 | grep -viE 'warn|converg'
echo "=== 8. layer sweep vs its own accuracy ==="
$APY run/layers.py runs/states_smolm.npz runs/smolm_gen.jsonl runs/layers_smolm.json 2>&1 \
    | grep -viE 'warn|converg|explained_var'
echo "=== 9. counterfactual control ==="
$PY run/cfprobe.py --model $M --data data/cal_smolm --probes runs/probes_smolm.npy \
    --out data/cf_smolm --res runs/cfprobe_smolm.json 2>&1 | grep -v "it/s" | tail -14
echo "=== STAGE12 DONE ==="
