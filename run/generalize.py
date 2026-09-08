# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Is the probe reading the attribute, or identifying the arm?

Each family holds 150 `canonical` items and 150 `anti` items, and the two arms were built with
**disjoint answer sets** -- chart canonical is multiples of 10 and anti is tens-plus-5, counting
is 2-5 against 8-12, spatial is left/right against above/below. Only tracking shares its label
set across arms.

That makes the pooled chance level wrong. Telling the arms apart is an easy visual
discrimination (round bars vs odd ones, few objects vs many, horizontal vs vertical relations),
and an item's arm halves or better its answer space for free. A probe scored against pooled
chance is credited for a discrimination that carries none of the attribute information the
claim is about.

So: fit and score the probe *inside each arm*, where the shortcut is unavailable, and compare
against the within-arm chance. The cross-arm transfer that would normally be the control here
cannot be run -- a probe trained on canonical can never emit an anti label -- and its 0% is an
artefact of that disjointness, not evidence. `arm_id` reports how separable the arms are, which
is the size of the shortcut being excluded.
"""
import json, sys
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

TARGET = dict(counting=lambda m: int(m["attribute"]["count"]),
              spatial=lambda m: m["attribute"]["relation"],
              chart=lambda m: int(m["attribute"]["value"]),
              tracking=lambda m: int(m["attribute"]["end"]))


def pipe(n):
    return make_pipeline(StandardScaler(),
                         PCA(n_components=min(64, n - 1), random_state=0),
                         LogisticRegression(max_iter=1000, C=0.5))


def cv(X, y, k=5):
    """Stratified k-fold inside one arm. Classes with <k members are dropped, and the share
    of items kept is reported so a high score on a thinned label set cannot pass unnoticed."""
    c = Counter(y); keep = np.array([i for i, v in enumerate(y) if c[v] >= k])
    if len(set(y[keep])) < 2: return float("nan"), 0.0, 0
    s = cross_val_score(pipe(len(keep)), X[keep], y[keep],
                        cv=StratifiedKFold(k, shuffle=True, random_state=0))
    return float(s.mean()), len(keep) / len(y), len(set(y[keep]))


def main(states="runs/states_3b.npz", branches="runs/branches6_test.jsonl",
         g1="runs/probe_g1.json", out="runs/generalize_3b.json"):
    npz = np.load(states); meta = json.load(open(states.replace(".npz", "_meta.json")))
    V = npz["vis"].astype(np.float32)
    G = json.load(open(g1))
    marm = {}
    for line in open(branches):
        r = json.loads(line); marm.setdefault((r["family"], r["arm"]), []).append(r["ok"]["none"])
    res = {}
    print(f"{'family':10s}{'L':>3s}  {'arm':10s}{'k':>3s}{'chance':>8s}{'probe':>8s}"
          f"{'model':>8s}{'lift':>8s}")
    for f in sorted(TARGET):
        l = G[f]["layer"]
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == f])
        y = np.array([TARGET[f](meta[i]) for i in idx]).astype(str)
        arm = np.array([meta[i]["arm"] for i in idx])
        X = V[idx, l]
        res[f] = dict(layer=int(l), arms={})
        for A in ["canonical", "anti"]:
            s = arm == A
            acc, kept, ncls = cv(X[s], y[s])
            ch = 1.0 / ncls if ncls else float("nan")
            m = float(np.mean(marm.get((f, A), [np.nan])))
            print(f"{f:10s}{l:3d}  {A:10s}{ncls:3d}{100*ch:7.0f}%{100*acc:7.0f}%"
                  f"{100*m:7.0f}%{100*(acc-m):+7.0f}")
            res[f]["arms"][A] = dict(probe=acc, chance=ch, n_classes=ncls, kept=kept, model=m)
        # how separable are the arms themselves -- the size of the shortcut excluded above
        a_acc, _, _ = cv(X, (arm == "anti").astype(int).astype(str))
        res[f]["arm_id"] = a_acc
        print(f"{'':13s}  {'arm_id':10s}{2:3d}{50:7.0f}%{100*a_acc:7.0f}%")
    print("\nprobe = 5-fold CV inside the arm, so arm identity carries no information.")
    print("chance = 1/n_classes within that arm. lift = probe - model, in points.")
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
