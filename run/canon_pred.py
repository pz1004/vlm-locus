# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Per-item predictions of the CANONICAL probe (layers.py, final layer, fixed hyperparameters).

layers.py reports accuracies only. Several analyses need the individual
predictions of that same estimator -- the value-band split, McNemar between probe and model, and
any per-item agreement between models -- and refitting them here, with the identical split seed
and pipeline, keeps them from silently coming from the other estimator.
"""
from __future__ import annotations
import json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sys.path.insert(0, "run")
from layers import TARGET
# the same filter layers.py applies, from the same definition: these predictions are what
# the paired test scores, so a filter that differed would compare two different item sets
from canon import MIN_CLASS, pairs_of, split


def fit(states, out):
    npz = np.load(states); meta = json.load(open(states.replace(".npz", "_meta.json")))
    V = npz["vis"].astype(np.float32); L = V.shape[1]; res = {}
    for f in sorted(TARGET):
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == f])
        if len(idx) == 0: continue
        y = np.array([TARGET[f](meta[i]) for i in idx])
        keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= MIN_CLASS], dtype=int)
        idx, y = idx[keep], y[keep]
        tr, _, te = split(y, groups=pairs_of(meta, idx), seed=0)
        pipe = make_pipeline(StandardScaler(), PCA(n_components=min(64, len(tr) - 1), random_state=0), LogisticRegression(max_iter=1000, C=0.5))
        pipe.fit(V[idx[tr], L - 1], y[tr])
        pred = pipe.predict(V[idx[te], L - 1])
        res[f] = dict(layer=L - 1, ids=[meta[i]["id"] for i in idx[te]], pred=[str(p) for p in pred], gold=[str(g) for g in y[te]], acc=float(np.mean(pred == y[te])))
        print(f"  {f:10s} n={len(te):3d}  final-layer acc {100*res[f]['acc']:5.1f}%")
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    fit(sys.argv[1], sys.argv[2])
