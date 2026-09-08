#!/bin/bash
# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
# LoRA at the same label budget as the probe, then the tracking re-sample of the sampling
# baselines. Serialised because both want the GPU.
set -e
.venv/bin/python run/lora.py --eval-base
bash run/stage8.sh
echo "=== STAGE9 DONE ==="
