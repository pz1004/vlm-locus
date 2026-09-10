# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Gate G1: does a per-instance evidence margin exist?

Linear probes only, fitted per (family, layer) on the residual stream at the last prompt
token, with three guards from the probing-methodology literature:

  * the baseline is not chance but the **same probe family fitted on blindfolded states**
    (Hewitt & Liang's logic applied to the vision case) -- decoding an attribute from a state
    that never saw the image would mean the probe is reading the question, not the evidence;
  * **selectivity** against a shuffled-label control -- a probe that fits noise as well as it
    fits the attribute is measuring capacity, not presence;
  * a **held-out split**, and the reported layer is the one selected on that split, not the
    maximum over layers on the test data.

The quantity LRC routes on is the per-item margin: log p_probe(true | visual state) minus
log p_probe(true | blindfolded state).
"""
import json, sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__))), "run"))
from canon import MIN_CLASS, split  # the filter is a protocol constant

npz = np.load(sys.argv[1] if len(sys.argv) > 1 else "runs/states_3b.npz")
meta = json.load(open((sys.argv[1] if len(sys.argv) > 1 else "runs/states_3b.npz")
                      .replace(".npz", "_meta.json")))
V, B = npz["vis"].astype(np.float32), npz["blind"].astype(np.float32)
L = V.shape[1]
fams = sorted({m["family"] for m in meta})
print(f"{len(meta)} items, {L} layers, dim {V.shape[2]}\n")

# the attribute each family's probe must recover -- the visual quantity, not the answer string
TARGET = dict(counting=lambda m: int(m["attribute"]["count"]),
              spatial=lambda m: m["attribute"]["relation"],
              chart=lambda m: int(m["attribute"]["value"]),
              tracking=lambda m: int(m["attribute"]["end"]),
              glyph=lambda m: int(m["attribute"]["value"]))

# 2048 raw dimensions against ~110 training items is not a linear probe, it is an
# interpolator: it would fit shuffled labels nearly as well as real ones. Standardise, project
# onto 64 PCA components fitted on the training split only, then regularise. That is the
# capacity control; the shuffled-label run below is what checks it worked.
def fit(X, y, Xt, yt, seed=0, k=64):
    clf = make_pipeline(StandardScaler(),
                        PCA(n_components=min(k, X.shape[0] - 1, X.shape[1]), random_state=seed),
                        LogisticRegression(max_iter=1000, C=0.5))
    clf.fit(X, y)
    P = clf.predict_proba(Xt)
    cls = list(clf.classes_)
    idx = [cls.index(v) if v in cls else -1 for v in yt]
    lp = np.array([np.log(max(P[i, j], 1e-12)) if j >= 0 else np.log(1e-12)
                   for i, j in enumerate(idx)])
    return clf.score(Xt, yt), lp

print(f"{'family':10s} {'layer':>6s} {'vis acc':>8s} {'blind acc':>10s} {'shuffled':>9s} "
      f"{'select.':>8s} {'chance':>7s} {'mean margin':>12s}  verdict")
out = {}
for f in fams:
    idx = np.array([i for i, m in enumerate(meta) if m["family"] == f])
    y = np.array([TARGET[f](meta[i]) for i in idx])
    keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= MIN_CLASS])   # drop rare classes
    idx, y = idx[keep], y[keep]
    tr, sel_i, te = split(y, seed=0)
    chance = max(np.bincount(np.unique(y, return_inverse=True)[1]).max() / len(y),
                 meta[idx[0]]["chance"])
    # Layer chosen on the SELECTION split, never on the test split. Taking the argmax layer
    # on test data is the uncorrected-maximum bug the protocol exists to avoid.
    best = None
    for l in range(0, L, 2):
        a, _ = fit(V[idx, l][tr], y[tr], V[idx, l][sel_i], y[sel_i])
        if best is None or a > best[1]: best = (l, a)
    l = best[0]
    Xv, Xb = V[idx, l], B[idx, l]
    av, lpv = fit(Xv[tr], y[tr], Xv[te], y[te])
    ab, lpb = fit(Xb[tr], y[tr], Xb[te], y[te])
    rng = np.random.default_rng(0); ysh = rng.permutation(y)
    ash, _ = fit(Xv[tr], ysh[tr], Xv[te], ysh[te])
    margin = lpv - lpb
    sel = av - ash
    # The gate must include the sign of the margin. A probe can be more accurate on visual
    # states while assigning the true class *lower* log-probability than the blindfolded probe
    # does -- better on average, worse calibrated, heavier tails. Such a probe passes an
    # accuracy-only gate and is still useless as a routing signal, because LRC routes on the
    # per-item margin, not on aggregate accuracy.
    ok = (av - ab > 0.05) and (sel > 0.10) and (margin.mean() > 0) and ((margin > 0).mean() > 0.75)
    print(f"{f:10s} {l:6d} {av:7.3f} {ab:9.3f} {ash:8.3f} {sel:7.3f} {chance:6.3f} "
          f"{margin.mean():+11.2f}  {'PASS' if ok else 'fail'}")
    out[f] = dict(layer=int(l), vis=float(av), blind=float(ab), shuffled=float(ash),
                  selectivity=float(sel), chance=float(chance),
                  mean_margin=float(margin.mean()), pos_margin=float((margin > 0).mean()),
                  n=int(len(idx)))
print()
print("G1 requires the visual probe to beat the blindfolded probe by >5 points AND to beat its")
print("own shuffled-label control by >10 points. A probe that passes only the first is reading")
print("the question; one that passes only the second is memorising.\n")
for f, d in out.items():
    print(f"  {f:10s} n={d['n']:3d}  items with margin>0: {100*d['pos_margin']:.0f}%")
json.dump(out, open((sys.argv[1] if len(sys.argv)>1 else "runs/states_3b.npz")
                    .replace("states_", "probe_g1_").replace(".npz", ".json"), "w"), indent=1)
