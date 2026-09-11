# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""The canonical probe's per-item predictions on the held-out split.

These feed the paired probe-vs-model test in run/canon.py, which is one of the four conditions
locus() requires.

This file replaces run/canon_pred.py, which did the same fit and produced byte-identical output.
That one took its output path as a command-line argument, so the string "canonpred" never
appeared in it, and it was called by no driver -- an artefact family whose producer no search for
its name could find. It was reported here as having no producer at all, which was wrong; what it
had was an unfindable one. The output path is a literal below for exactly that reason, and
run/verify_protocol.py asserts that every artefact family the analysis reads is named in some
tracked producer.

What this adds over the file it replaces is pinned BLAS threads, so a reader on another machine
gets the same predictions, and --check, so the committed files can be shown to be reproducible
rather than assumed to be.

The fit is run/layers.py's, at the final layer, and must stay that way: same rare-class filter,
same pair-grouped split at seed 0, same pipeline and hyperparameters. It is not re-derived here.
layers.py is imported and its own objects are used, so this cannot drift into a second definition
of "the canonical probe" -- which is the failure mode the rest of this protocol keeps finding.

    python3 run/canonpred.py --all
    python3 run/canonpred.py --tag realchart_v2 --check
"""
from __future__ import annotations
import os
# Pinned before numpy loads, for run/nullcal.py's reason: the reduction order inside a fit
# depends on the thread count, and it flips the occasional tied prediction. These files are
# compared byte for byte against the committed ones, so a tie that lands differently is the
# difference between "reproduced" and "changed a published p-value".
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from canon import MIN_CLASS, GEN, pairs_of, split
from layers import TARGET          # one definition of what each family's probe decodes


def predict(tag):
    """Final-layer canonical probe, per item, on the held-out split."""
    st = f"runs/states_{tag}.npz"
    if not os.path.exists(st):
        return None
    npz = np.load(st)
    meta = json.load(open(st.replace(".npz", "_meta.json")))
    V = npz["vis"].astype(np.float32)
    L = V.shape[1]
    out = {}
    for f in sorted(TARGET):
        idx = np.array([i for i, m in enumerate(meta) if m["family"] == f])
        if len(idx) == 0:
            continue
        y = np.array([TARGET[f](meta[i]) for i in idx])
        keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= MIN_CLASS], dtype=int)
        idx, y = idx[keep], y[keep]
        tr, _, te = split(y, groups=pairs_of(meta, idx), seed=0)
        pipe = make_pipeline(StandardScaler(),
                             PCA(n_components=min(64, len(tr) - 1), random_state=0),
                             LogisticRegression(max_iter=1000, C=0.5))
        pipe.fit(V[idx[tr], L - 1], y[tr])
        pred = pipe.predict(V[idx[te], L - 1])
        out[f] = dict(layer=int(L - 1),
                      ids=[meta[i]["id"] for i in idx[te]],
                      pred=[str(v) for v in pred],
                      gold=[str(v) for v in y[te]],
                      acc=float(np.mean(pred == y[te])))
    return out


def tags():
    """Exactly the cells canon.py reads a paired test for -- its grid plus the control.

    Not every tag with captured states: globbing runs/states_*.npz also picks up superseded
    captures, and writing a canonpred beside a quarantined tag puts a live-looking artefact next
    to one the provenance check exists to keep out of the load path.
    """
    from canon import REAL, SYNTH, BIDIR
    return [t for t, _, _ in list(SYNTH) + list(REAL) + list(BIDIR)]


def main(a):
    todo = [a.tag] if a.tag else tags()
    bad = 0
    for t in todo:
        d = predict(t)
        if d is None:
            print(f"  {t:30s} no states, skipped")
            continue
        p = f"runs/canonpred_{t}.json"
        new = json.dumps(d, indent=1)
        if a.check:
            if not os.path.exists(p):
                print(f"  {t:30s} MISSING on disk")
                bad += 1
                continue
            old = json.load(open(p))
            same = all(old.get(f, {}).get(k) == d[f][k]
                       for f in d for k in ("layer", "ids", "pred", "gold"))
            acc = all(abs(old.get(f, {}).get("acc", -1) - d[f]["acc"]) < 1e-12 for f in d)
            print(f"  {t:30s} {'reproduces' if same and acc else 'DIFFERS'}"
                  f"   families {sorted(d)}")
            bad += not (same and acc)
        else:
            open(p, "w").write(new + "\n")
            print(f"  {t:30s} wrote {p}  families {sorted(d)}")
    if a.check:
        print(f"\n{'all reproduce' if not bad else str(bad) + ' tag(s) differ'}")
    return bad


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", help="one tag; default is every tag with captured states")
    p.add_argument("--all", action="store_true", help="explicit form of the default")
    p.add_argument("--check", action="store_true",
                   help="compare against the committed files instead of writing")
    sys.exit(main(p.parse_args()))
