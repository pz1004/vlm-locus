# SPDX-FileCopyrightText: 2026 Sooyoung Jang
# SPDX-License-Identifier: GPL-3.0-only
"""Calibrate the false-positive rate of the presence verdict, per cell, at the final layer.

The protocol issues a presence verdict when the probe reads the attribute out of the final
hidden state better than the model emits it. That verdict needs a null: how high can the
reported probe accuracy go when the representation carries *no* usable information about the
attribute? Until now the answer was a single constant (2.5 pp, hardcoded in run/canon.py and
run/p3.py) priced once on the designed-absence family and applied to every cell.

A constant cannot serve. The null moves with the number of classes and the sample size, and
those vary by a factor of five across our grid, so one threshold is simultaneously too strict
on 4-class families and too permissive on 21-class ones.

Two different permutation quantities are computed here, and they answer different questions:

  selectivity (Hewitt & Liang control task).  Train on permuted labels, score against the
  *same* permuted labels. Measures the probe family's capacity to fit an arbitrary relabelling
  of a fixed representation. Used to choose probe complexity. run/probe.py already reports it
  as `shuffled`, from a single draw.

  null (this file).  Train on permuted labels, score against the *true* test labels. Measures
  how high the reported accuracy can go with the label-representation relation destroyed --
  i.e. the false-positive rate of this cell's presence verdict. This is a decision threshold, not a capacity measure, and it is what replaces the constant.

The estimator, split and rare-class filter mirror run/layers.py exactly, so the null is
calibrated for the same probe that produces the reported number. Reads the cached activations
in runs/states_*.npz, so it needs no GPU.

Also computed at the final layer, so that G1 can be evaluated where the probe is read
rather than at a selected layer: a refitted blindfold probe and the per-item margin statistics
that run/probe.py's gate requires and run/canon.py currently drops.
"""
from __future__ import annotations
import os
# Pin BLAS threads before numpy loads. Without this the number of parallel workers changes the
# floating-point reduction order inside each fit, which flips the occasional tied prediction and
# moves the empirical p by a draw or two -- small, but it makes the null irreproducible.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob, json, sys, time, warnings
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# the rare-class filter is a protocol constant, not a local choice: gen/build_chart.py
# uses it to decide which edit targets a probe will be able to emit, so a copy here that
# drifted would produce counterfactuals no probe could follow. One definition, imported.
from canon import MIN_CLASS, pairs_of, split

TARGET = dict(counting=lambda m: int(m["attribute"]["count"]), spatial=lambda m: m["attribute"]["relation"], chart=lambda m: int(m["attribute"]["value"]), tracking=lambda m: int(m["attribute"]["end"]), glyph=lambda m: int(m["attribute"]["value"]))

# a permuted fit can hand PCA a degenerate component; the draw is still valid
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

NPERM = int(os.environ.get("VLM_LOCUS_NPERM", 2000))
NJOBS = int(os.environ.get("VLM_LOCUS_NJOBS", 24))


def pipe(k, n, seed=0):
    return make_pipeline(StandardScaler(), PCA(n_components=min(k, n - 1), random_state=seed), LogisticRegression(max_iter=1000, C=0.5))


def fit(X, y, Xt, yt):
    """Accuracy plus the log-probability the probe assigns the true class, item by item."""
    clf = pipe(64, X.shape[0]).fit(X, y)
    P, cls = clf.predict_proba(Xt), list(clf.classes_)
    j = [cls.index(v) if v in cls else -1 for v in yt]
    lp = np.array([np.log(max(P[i, c], 1e-12)) if c >= 0 else np.log(1e-12)
                   for i, c in enumerate(j)])
    return float(clf.score(Xt, yt)), lp


def cell(V, B, idx, y, tr, te):
    Xv, Xb = V[idx, -1], B[idx, -1]
    av, lpv = fit(Xv[tr], y[tr], Xv[te], y[te])
    ab, lpb = fit(Xb[tr], y[tr], Xb[te], y[te])          # refitted, as run/probe.py does
    at = float(pipe(64, len(tr)).fit(Xv[tr], y[tr]).score(Xb[te], y[te]))   # transfer, as layers.py
    rng = np.random.default_rng(0)
    ysh = rng.permutation(y)
    ash, _ = fit(Xv[tr], ysh[tr], Xv[te], ysh[te])        # control-task selectivity, one draw

    # the null: train on noise, score against the truth
    def draw(s):
        yy = np.random.default_rng(1000 + s).permutation(y[tr])
        return float(pipe(64, len(tr)).fit(Xv[tr], yy).score(Xv[te], y[te]))
    null = np.array(joblib.Parallel(n_jobs=NJOBS)(joblib.delayed(draw)(s) for s in range(NPERM)))

    m = lpv - lpb
    return dict(n_train=int(len(tr)), n_test=int(len(te)), classes=int(len(set(y.tolist()))), vis=av, blind_refit=ab, blind_transfer=at, shuffled=ash, selectivity=av - ash, mean_margin=float(m.mean()), pos_margin=float((m > 0).mean()), null_mean=float(null.mean()), null_sd=float(null.std()), null_q95=float(np.quantile(null, .95)), null_q99=float(np.quantile(null, .99)), null_max=float(null.max()), nperm=NPERM, # the draws are k/n_test, so the distribution is exactly summarised by its
                # value counts -- enough to plot, and far smaller than 2000 floats per cell
                null_hist={f"{v:.6f}": int(c) for v, c in
                           zip(*[x.tolist() for x in np.unique(null, return_counts=True)])}, # empirical one-sided p, add-one corrected so it can never be exactly zero
                p_null=float((1 + (null >= av).sum()) / (1 + NPERM)), # G1 at the final layer, under the full gate run/probe.py defines
                g1=bool(av - ab > 0.05 and av - ash > 0.10
                        and m.mean() > 0 and (m > 0).mean() > 0.75))


def main(tags):
    for tag in tags:
        st = f"runs/states_{tag}.npz"
        if not os.path.exists(st):
            print(f"  skip {tag}: no {st}"); continue
        npz = np.load(st)
        meta = json.load(open(st.replace(".npz", "_meta.json")))
        V, B = npz["vis"].astype(np.float32), npz["blind"].astype(np.float32)
        out, t0 = {}, time.time()
        print(f"\n{tag}  ({V.shape[0]} items, {V.shape[1]} layers, dim {V.shape[2]}, "
              f"{NPERM} perms)")
        print(f"  {'family':10s} {'ncls':>5s} {'vis':>7s} {'blind':>7s} {'shuf':>7s} "
              f"{'null mu':>8s} {'q95':>7s} {'max':>7s} {'p':>7s}  G1")
        for f in sorted(TARGET):
            idx = np.array([i for i, m in enumerate(meta) if m["family"] == f])
            if len(idx) == 0: continue
            y = np.array([TARGET[f](meta[i]) for i in idx])
            keep = np.array([c for c in range(len(y)) if (y == y[c]).sum() >= MIN_CLASS], dtype=int)
            idx, y = idx[keep], y[keep]
            tr, _, te = split(y, groups=pairs_of(meta, idx), seed=0)
            r = cell(V, B, idx, y, tr, te)
            r["chance"] = float(meta[idx[0]]["chance"])
            out[f] = r
            print(f"  {f:10s} {r['classes']:5d} {100*r['vis']:6.1f}% {100*r['blind_refit']:6.1f}% "
                  f"{100*r['shuffled']:6.1f}% {100*r['null_mean']:7.1f}% {100*r['null_q95']:6.1f}% "
                  f"{100*r['null_max']:6.1f}% {r['p_null']:7.4f}  {'PASS' if r['g1'] else 'fail'}")
        json.dump(out, open(f"runs/null_{tag}.json", "w"), indent=1)
        print(f"  wrote runs/null_{tag}.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    tags = sys.argv[1:] or [os.path.basename(p)[len("layers_"):-len(".json")]
                            for p in sorted(glob.glob("runs/layers_*.json"))]
    main(tags)
