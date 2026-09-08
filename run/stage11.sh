#!/bin/bash
# The comparison that decides the claim: LoRA and B-read at the *same* label budget.
# At the full split LoRA wins (81.2% vs 75.2%). run/datasize.py shows the probe passing the
# model at 20 labels per family, so the question is what fine-tuning does with the same 20.
# Every epoch that is evaluated is a chance for LoRA to look better -- the best epoch is
# reported, which is oracle model selection and generous to the competitor.
set -e
PY=${PY:-.venv/bin/python}
echo "=== LoRA, 20 labels/family (80 items), 12 epochs ==="
$PY run/lora.py --n-per-family 20 --epochs 12 --eval-every 4 \
    --tag lora20 --out runs/lora20_3b.json 2>&1 | grep -viE 'warn|it/s|Loading|use_cache|detach'
echo "=== LoRA, 40 labels/family (160 items), 8 epochs ==="
$PY run/lora.py --n-per-family 40 --epochs 8 --eval-every 4 \
    --tag lora40 --out runs/lora40_3b.json 2>&1 | grep -viE 'warn|it/s|Loading|use_cache|detach'
echo "=== STAGE11 DONE ==="
