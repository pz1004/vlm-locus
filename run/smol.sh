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
M=HuggingFaceTB/SmolVLM-Instruct
echo "=== SmolVLM: difficulty sweep on the shared level-swept set ==="
$PY run/score.py --model $M --data data/sw4 --out runs/sw4_smol.jsonl --base-only --gen-only
python3 - <<'PY'
import json, numpy as np
rs=[json.loads(l) for l in open('runs/sw4_smol.jsonl')]
best={}
print(f"{'family':10s} " + ''.join(f'{f"L{l}":>9s}' for l in range(5)) + '   -> lock')
for f in sorted({r['family'] for r in rs}):
    row=[100*(1-np.mean([x['gen_correct'] for x in rs if x['family']==f and x['level']==l])) for l in range(5)]
    cand=[l for l in range(5) if 40<=row[l]<=60]
    pick=cand[len(cand)//2] if cand else int(np.argmin([abs(e-50) for e in row]))
    best[f]=pick
    print(f'{f:10s} ' + ''.join(f'{e:8.0f}%' for e in row) + f'   L{pick} ({row[pick]:.0f}%)')
json.dump(best, open('runs/levels_smol.json','w'), indent=1)
PY
LV=$(cat runs/levels_smol.json | tr -d '\n ')
echo "=== SmolVLM: generate at its own locked levels $LV ==="
rm -rf data/cal_smol
python3 gen/generate.py --out data/cal_smol --n 300 --seed 20260906 --no-cf --levels "$LV" | tail -1
echo "=== SmolVLM: behaviour ==="
$PY run/score.py --model $M --data data/cal_smol --out runs/cal_smol_gen.jsonl --base-only --gen-only
echo "=== SmolVLM: hidden states ==="
$PY run/capture.py --model $M --data data/cal_smol --out runs/states_smol.npz
echo "=== SmolVLM: G1 ==="
$APY run/probe.py runs/states_smol.npz 2>&1 | grep -viE 'warn|converg'
echo "=== SmolVLM: locus split ==="
$PY run/locus.py runs/states_smol.npz runs/cal_smol_gen.jsonl runs/probe_g1_smol.json 2>&1 | grep -viE 'warn|converg'
echo "=== SMOL DONE ==="
